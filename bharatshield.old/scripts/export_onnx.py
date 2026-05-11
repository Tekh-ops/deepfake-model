# scripts/export_onnx.py

import torch

def export_spatial_model(checkpoint_path: str, output_path: str):
    model = SpatialDetector(model_name='efficientnet_b4')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval().cuda()
    
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['logit', 'embedding'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logit': {0: 'batch_size'}
        }
    )
    print(f"Exported to {output_path}")
    
    # Verify the export
    import onnxruntime as ort
    sess = ort.InferenceSession(
        output_path,
        providers=['CUDAExecutionProvider']
    )
    out = sess.run(None, {'image': dummy_input.cpu().numpy()})
    print(f"ONNX output shape: {out[0].shape}")