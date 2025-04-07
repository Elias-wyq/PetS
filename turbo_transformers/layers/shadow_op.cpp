#include "turbo_transformers/layers/shadow_op.h"
#include "turbo_transformers/layers/kernels/sparse_mat_mul.h"
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/utils.h"
#include "turbo_transformers/layers/kernels/fused_adapter_kernel.h"

namespace turbo_transformers {
namespace layers {

void compute_mask_shadow(
            const core::PETLayerManager& pet_layer_manager,
            const int task_id,
            core::Tensor* task_input,
            core::Tensor* task_output, 
            core::Tensor* task_shadow_output,
            core::Tensor* task_hidden_states,
            core::Tensor* task_q_out,
            core::Tensor* task_k_out,
            core::Tensor* task_v_out,

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name){
    // Compute the shadow output    
    core::Tensor * operand_A;

    if(task_hidden_states){
        operand_A = task_hidden_states;
    }
    else{
        operand_A = task_input;
    }

    auto cuda_ctx =
        turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

#ifdef SP_SHADOW
    std::shared_ptr<kernels::SparseMatMulCsr> sparse_matmul_ptr = pet_layer_manager.get_sparse_matmul(task_id);
    core::SparseTensor* sp_task_mask_ptr = pet_layer_manager.get_sp_mask_shadow(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_mask_ptr, false,
          1.0, task_shadow_output, 0.0, cuda_ctx.get(), name + "SparseMatMul");
    } else {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_mask_ptr, false,
          1.0, task_shadow_output, 0.0, name + "SparseMatMul");
    }
#else
    // get the mask shadow
    const core::Tensor &task_mask_shadow = pet_layer_manager.get_maskbert_shadow(task_id); 
    //task shadow output
    layers::kernels::MatMul(*operand_A, false, task_mask_shadow, false, 1.0,
                            task_shadow_output, 0.0, name + "/MASK/MatMul");
#endif

    // Add to the dense output
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               cuda_ctx.get(),
                               name + "/MASK/ElwsAdd");
    } else {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               name + "/MASK/ElwsAdd");
    }

    const core::Tensor &task_bias = pet_layer_manager.get_bias(-1);
    if(add_input_bias_layernorm){
        const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(-1);
        const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(-1);
        if (core::CUDADeviceContext::num_streams > 1) {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, cuda_ctx.get(), 1e-12, name + "/MASK/AddBiasLayerNorm");
        } else {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, 1e-12, name + "/MASK/AddBiasLayerNorm");
        }
    }
    else if(add_bias_act){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, cuda_ctx.get(), name + "MASK/AddBiasAct");
      } else {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, name + "MASK/AddBiasAct");
      }
    }
    else if(split_add_transpose){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::SplitAddBiasTransposeForScore(
            *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
            cuda_ctx.get(),
            name + "/SplitAddBiasTransposeForScore");
      } else {
        layers::kernels::SplitAddBiasTransposeForScore(
            *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
            name + "/SplitAddBiasTransposeForScore");
      }
    }
}

void compute_diff_shadow(
            const core::PETLayerManager& pet_layer_manager,
            const int task_id,
            core::Tensor* task_input,
            core::Tensor* task_output, 
            core::Tensor* task_shadow_output,
            core::Tensor* task_hidden_states,
            core::Tensor* task_q_out,
            core::Tensor* task_k_out,
            core::Tensor* task_v_out,

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name){
    // Compute the shadow output
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

    core::Tensor * operand_A;
    if(task_hidden_states){
        operand_A = task_hidden_states;
    }
    else{
        operand_A = task_input;
    }
#ifdef SP_SHADOW
    std::shared_ptr<kernels::SparseMatMul> sparse_matmul_ptr = pet_layer_manager.get_sparse_matmul(task_id);
    core::SparseTensor* sp_task_diff_ptr = pet_layer_manager.get_sp_diff_shadow(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
    sparse_matmul_ptr->Run(
        *operand_A, false,
        sp_task_diff_ptr, false,
        1.0, task_shadow_output, 0.0, cuda_ctx.get(), name + "SparseMatMul");
    } else {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_diff_ptr, false,
          1.0, task_shadow_output, 0.0, name + "SparseMatMul");
    }
#else 
    const core::Tensor &task_diff_shadow = pet_layer_manager.get_diff_shadow(task_id); // get the mask shadow  
    //task shadow output
    layers::kernels::MatMul(*operand_A, false, task_diff_shadow, false, 1.0,
            task_shadow_output, 0.0, name + "/DIFF/MatMul");
#endif
    // Add to the dense output
    const core::Tensor &task_bias = pet_layer_manager.get_bias(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               cuda_ctx.get(),
                               name + "/DIFF/ElwsAdd");
    } else {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               name + "/DIFF/ElwsAdd");
    }
    if(add_input_bias_layernorm) {
        const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
        const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
        if (core::CUDADeviceContext::num_streams > 1) {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, cuda_ctx.get(), 1e-12, name + "/DIFF/AddBiasLayerNorm");
        } else {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, 1e-12, name + "/DIFF/AddBiasLayerNorm");
        }
    }
    else if(add_bias_act){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, cuda_ctx.get(), name + "DIFF/AddBiasAct");
      } else {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, name + "DIFF/AddBiasAct");
      }
    }
    else if(split_add_transpose){
      if (core::CUDADeviceContext::num_streams > 1) {
       layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     cuda_ctx.get(),
                                                     name + "/SplitAddBiasTransposeForScore");
      } else {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     name + "/SplitAddBiasTransposeForScore");
      }
    }
}

void compute_bitfit_shadow(
        const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out, 
        bool add_bias_act, bool add_input_bias_layernorm,
        bool split_add_transpose, std::string&name){
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &task_bias = pet_layer_manager.get_bias(task_id);
  if(add_input_bias_layernorm){
    const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
    const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
          task_output, cuda_ctx.get(), 1e-12, name + "/BITFIT/AddBiasLayerNorm");
    } else {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
          task_output, 1e-12, name + "/BITFIT/AddBiasLayerNorm");
    }
  }
  else if(add_bias_act) {
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, cuda_ctx.get(), name + "BITFIT/AddBiasAct");
    } else {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, name + "BITFIT/AddBiasAct");
    }
  }
  else if(split_add_transpose){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     cuda_ctx.get(),
                                                     name + "/SplitAddBiasTransposeForScore");
    } else {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     name + "/SplitAddBiasTransposeForScore");
    }
  }
}

void compute_adapter_shadow(
        const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out, 
        bool add_bias_act, bool add_input_bias_layernorm,
        bool split_add_transpose, std::string&name){
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &down_scale_w = pet_layer_manager.get_adapter_params(task_id, true, true);
  const core::Tensor &down_scale_b = pet_layer_manager.get_adapter_params(task_id, true, false);
  const core::Tensor &up_scale_w = pet_layer_manager.get_adapter_params(task_id, false, true);
  const core::Tensor &up_scale_b = pet_layer_manager.get_adapter_params(task_id, false, false);
    
  // 使用融合的Adapter内核
  if (core::CUDADeviceContext::num_streams > 1) {
    // 获取输入和输出的指针和维度
    const float* input_ptr = task_output->data<float>();
    float* output_ptr = task_output->mutableData<float>();
    
    int batch_size = task_output->shape(0);
    int seq_len = task_output->shape(1);
    int hidden_size = task_output->shape(2);
    int intermediate_size = down_scale_b.shape(0);
    
    // 调用融合的Adapter内核
    kernels::LaunchFusedAdapterKernel(
        input_ptr,
        down_scale_w.data<float>(),
        down_scale_b.data<float>(),
        up_scale_w.data<float>(),
        output_ptr,
        batch_size,
        seq_len,
        hidden_size,
        intermediate_size,
        cuda_ctx.get()->stream());
  } else {
    // 单流版本也使用融合内核
    const float* input_ptr = task_output->data<float>();
    float* output_ptr = task_output->mutableData<float>();
    
    int batch_size = task_output->shape(0);
    int seq_len = task_output->shape(1);
    int hidden_size = task_output->shape(2);
    int intermediate_size = down_scale_b.shape(0);
    
    // 调用融合的Adapter内核
    kernels::LaunchFusedAdapterKernel(
        input_ptr,
        down_scale_w.data<float>(),
        down_scale_b.data<float>(),
        up_scale_w.data<float>(),
        output_ptr,
        batch_size,
        seq_len,
        hidden_size,
        intermediate_size,
        cudaStreamDefault);
  }
  
  // 处理LayerNorm (如果需要)
  if(add_input_bias_layernorm){
    const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
    const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, up_scale_b, task_layer_norm_weight, task_layer_norm_bias,
          task_output, cuda_ctx.get(), 1e-12, name + "/Adapter/AddBiasLayerNorm");
    } else {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, up_scale_b, task_layer_norm_weight, task_layer_norm_bias,
          task_output, 1e-12, name + "/Adapter/AddBiasLayerNorm");
    }
  }
  else if(add_bias_act){
    std::cerr << "Invalid operation" << std::endl;
  }
  else if(split_add_transpose){
    std::cerr << "Invalid operation" << std::endl;
  }
}

void compute_nothing(const core::PETLayerManager& pet_layer_manager,
                     const int task_id,
                     core::Tensor* task_input,
                     core::Tensor* task_output, 
                     core::Tensor* task_shadow_output,
                     core::Tensor* task_hidden_states,
                     core::Tensor* task_q_out,
                     core::Tensor* task_k_out,
                     core::Tensor* task_v_out, 
                     bool add_bias_act,  bool add_input_bias_layernorm,
                     bool split_add_transpose, std::string&name) {
  auto cuda_ctx =
        turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &task_bias = pet_layer_manager.get_bias(-1);
  if(add_input_bias_layernorm){
      const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(-1);
      const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(-1);
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasLayerNorm<float>(
            *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
            task_output, cuda_ctx.get(), 1e-12, name + "/Nothing/AddBiasLayerNorm");
      } else {
        layers::kernels::AddBiasLayerNorm<float>(
            *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
            task_output, 1e-12, name + "/Nothing/AddBiasLayerNorm");
      }
  }
  else if(add_bias_act){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, cuda_ctx.get(), name + "Nothing/AddBiasAct");
    } else {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, name + "Nothing/AddBiasAct");
    }
  }
  else if(split_add_transpose){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::SplitAddBiasTransposeForScore(
          *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
          cuda_ctx.get(),
          name + "/SplitAddBiasTransposeForScore");
    } else {
      layers::kernels::SplitAddBiasTransposeForScore(
          *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
          name + "/SplitAddBiasTransposeForScore");
    }
  }

};

shadow_op get_shadow_op(
  const core::PETLayerManager& pet_layer_manager,
  int task_id){
    int pet_type = pet_layer_manager.get_pet_type(task_id);

    switch (pet_type)
    {
    case MASK_BERT:
        return compute_mask_shadow;
    case DIFF_PRUNING:
        return compute_diff_shadow;
    case BITFIT:
        return compute_bitfit_shadow;
    case ADAPTERS:
        return compute_adapter_shadow;
    case STANDARD:
        return compute_nothing;
    default:
        std::cerr<<"Unsupported Shadow Operation!"<<std::endl;
        // 默认返回compute_nothing作为fallback
        return compute_nothing;
    }
}

} // namespace layers
} // namespace turbo_transformers
