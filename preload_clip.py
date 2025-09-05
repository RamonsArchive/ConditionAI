#!/usr/bin/env python3
"""
Pre-load CLIP model at Docker build time for faster startup
"""
import torch
import clip
import os
import sys

def preload_clip_models():
    """Pre-load CLIP models and cache them"""
    print('🚀 Pre-loading CLIP models at build time...')
    
    models_loaded = []
    
    # Try to load primary model (ViT-L/14)
    try:
        print('📥 Loading CLIP ViT-L/14 (primary model)...')
        model, preprocess = clip.load('ViT-L/14', device='cpu')
        print('✅ CLIP ViT-L/14 loaded and cached successfully!')
        models_loaded.append('ViT-L/14')
        
        # Test the model works
        test_text = clip.tokenize(["a test item"])
        with torch.no_grad():
            text_features = model.encode_text(test_text)
            print(f'✅ ViT-L/14 test passed, output shape: {text_features.shape}')
            
    except Exception as e:
        print(f'⚠️ ViT-L/14 loading failed: {e}')
    
    # Try to load fallback model (ViT-B/32)
    try:
        print('📥 Loading CLIP ViT-B/32 (fallback model)...')
        model_fallback, preprocess_fallback = clip.load('ViT-B/32', device='cpu')
        print('✅ CLIP ViT-B/32 loaded and cached successfully!')
        models_loaded.append('ViT-B/32')
        
        # Test the fallback model works
        test_text = clip.tokenize(["a test item"])
        with torch.no_grad():
            text_features = model_fallback.encode_text(test_text)
            print(f'✅ ViT-B/32 test passed, output shape: {text_features.shape}')
            
    except Exception as e:
        print(f'⚠️ ViT-B/32 loading failed: {e}')
    
    # Check if we loaded at least one model
    if not models_loaded:
        print('❌ No CLIP models could be loaded!')
        return False
    
    # Save verification data
    verification_data = {
        'models_loaded': models_loaded,
        'primary_model': 'ViT-L/14' in models_loaded,
        'fallback_model': 'ViT-B/32' in models_loaded,
        'build_time': True,
        'available_models': clip.available_models()
    }
    
    torch.save(verification_data, '/app/clip_verification.pt')
    print('✅ CLIP verification file saved')
    print(f'📋 Models cached: {models_loaded}')
    
    return True

if __name__ == '__main__':
    try:
        success = preload_clip_models()
        if not success:
            print('❌ CLIP preloading failed completely')
            sys.exit(1)
        print('🎯 CLIP preloading completed successfully!')
    except Exception as e:
        print(f'❌ CLIP preloading failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)