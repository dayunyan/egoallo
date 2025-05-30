import pickle
import numpy as np
from pathlib import Path

def validate_hamer_outputs(outputs):
    """验证SavedHamerOutputs对象的结构完整性"""
    checks_passed = True

    # 检查基本属性是否存在
    required_attrs = [
        'mano_faces_left', 'mano_faces_right',
        'detections_left_wrt_cam', 'detections_right_wrt_cam',
        'T_device_cam', 'T_cpf_cam'
    ]
    outputs_keys = outputs.keys()
    for attr in required_attrs:
        if not attr in outputs_keys:
            print(f"❌ 缺失必要属性: {attr}")
            checks_passed = False

    # 验证MANO面数据
    for side in ['left', 'right']:
        faces = outputs[f'mano_faces_{side}']
        if not isinstance(faces, np.ndarray):
            print(f"❌ {side}_faces类型错误: {type(faces)}，应为np.ndarray")
            checks_passed = False
        elif faces.shape[1] != 3:
            print(f"❌ {side}_faces形状异常: {faces.shape}，应为(N,3)")
            checks_passed = False

    # 验证检测字典
    for hand in ['left', 'right']:
        detections = outputs[f'detections_{hand}_wrt_cam']
        if not isinstance(detections, dict):
            print(f"❌ {hand}_detections类型错误: {type(detections)}，应为dict")
            checks_passed = False
            continue
        
        # 抽样检查检测项
        for ts, detection in list(detections.items())[:3]:
            if detection is None:
                continue
            required_keys = ['verts', 'keypoints_3d', 'mano_hand_pose', 'mano_hand_betas']
            for key in required_keys:
                if key not in detection:
                    print(f"❌ {hand}检测项缺少键: {key}")
                    checks_passed = False

    # 验证变换矩阵
    for transform in ['T_device_cam', 'T_cpf_cam']:
        t = outputs[transform]
        if not isinstance(t, np.ndarray):
            print(f"❌ {transform}类型错误: {type(t)}，应为np.ndarray")
            checks_passed = False
        elif t.shape != (7,):
            print(f"❌ {transform}形状异常: {t.shape}，应为(7,)")
            checks_passed = False

    return checks_passed

def inspect_hamer_pkl(pkl_path: Path):
    """解析并检查pkl文件"""
    try:
        with open(pkl_path, 'rb') as f:
            outputs = pickle.load(f)
            print(type(outputs), outputs.keys())
            
        print("✅ 文件加载成功")
        print("\n=== 基本属性检查 ===")
        if validate_hamer_outputs(outputs):
            print("\n✅ 所有结构检查通过")
        else:
            print("\n⚠️ 发现结构问题，请检查上述警告")

        print("\n=== 数据摘要 ===")
        print(f"左手检测数量: {len(outputs['detections_left_wrt_cam'])}")
        print(f"右手检测数量: {len(outputs['detections_right_wrt_cam'])}")
        print(f"左手检测数据样例：{outputs['detections_left_wrt_cam'].keys()}")
        print(f"MANO左面数: {outputs['mano_faces_left'].shape[0]}")
        print(f"MANO右面数: {outputs['mano_faces_right'].shape[0]}")
        print(f"MANO左面数据样例：{outputs['mano_faces_left'][0]}")
        # print(f"设备到相机变换: {outputs['T_device_cam'][:3]}... (平移部分)")

        # 显示第一个有效检测
        print("\n=== 示例检测数据 ===")
        for hand in ['left', 'right']:
            detections = outputs[f'detections_{hand}_wrt_cam']
            first_valid = next((d for d in detections.values() if d is not None), None)
            if first_valid:
                print(f"第一个有效{hand}手检测:")
                print(f"  顶点形状: {first_valid['verts'].shape}")
                print(f"  关键点形状: {first_valid['keypoints_3d'].shape}")
                break

    except FileNotFoundError:
        print(f"❌ 文件不存在: {pkl_path}")
    except Exception as e:
        print(f"❌ 解析失败: {str(e)}")

if __name__ == "__main__":
    pkl_path = Path("/dfs/data/Datasets/EgoExoBodyPose/takes_image_downscaled_448/cmu_bike01_2/hamer_outputs.pkl")
    inspect_hamer_pkl(pkl_path)