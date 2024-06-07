# 3D Image Processing

## 1 D455에 대해
Intel RealSense Depth Camera D455는 Intel에서 제작한 고급 3D 깊이 카메라로, 다양한 응용 분야에서 높은 정확도와 정밀도를 제공합니다. 이 카메라는 특히 로봇 공학, 드론, 증강 현실(AR), 가상 현실(VR), 산업 자동화, 3D 스캐닝 등에서 많이 사용됩니다.

- 깊이 센서: 듀얼 오버헤드 IR 카메라, 스테레오 이미지 센서
- RGB 카메라: 1280x720 해상도, 30fps
(명심할 것, RGB 카메라의 해상도와 Depth 카메라의 해상도는 다르다.)

- 최대 깊이 거리: 6미터
- 베이스라인: 95mm
- FOV(시야각): 86° × 57° (깊이), 90° × 65° (RGB)
- IMU: 6자유도 가속도계 및 자이로스코프
- 연결: USB 3.1 Gen 1


## 2. (중요) D455의 RGB 해상도와 Depth 해상도

1. RGB 카메라 해상도 범위:
- 1920x1080 @ 30fps
- 1280x720 @ 30fps, 60fps
- 960x540 @ 60fps, 90fps
- 848x480 @ 60fps, 90fps
- 640x480 @ 60fps, 90fps
- 640x360 @ 60fps, 90fps
- 424x240 @ 60fps, 90fps
- 320x240 @ 60fps, 90fps
- 320x180 @ 60fps, 90fps
  
2. 깊이 카메라 해상도 범위:
- 1280x800 @ 30fps
- 1280x720 @ 30fps, 15fps
- 848x480 @ 90fps, 60fps, 30fps
- 640x480 @ 90fps, 60fps, 30fps
- 640x360 @ 60fps, 30fps
- 480x270 @ 60fps, 30fps
- 424x240 @ 90fps, 60fps, 30fps

## 3. 컴퓨터로 촬영 후, 저장

다음과 같은 방법으로 촬영하고 저장할 수 있다.

```
import cv2
from pyk4a import PyK4A
import numpy as np
import matplotlib.pyplot as plt

def take_picture_print(capture, name1, name2, name3):

    img_depth = capture.depth
    img_color = capture.color
    img_ir = capture.ir

    img_rgb = img_color[:, :, 2::-1]

    print(f"depth: {img_depth.shape}, rgb: {img_rgb.shape}, ir: {img_ir.shape}")

    cv2.imwrite(name1, img_depth)
    cv2.imwrite(name2, img_rgb)
    cv2.imwrite(name3, img_ir)

    plt.figure(figsize = (12,8))
    plt.subplot(1,2,1)
    plt.imshow(img_depth, cmap = 'gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.show()

    plt.imshow(img_ir)
    plt.axis('off')
    plt.show()

k4a = PyK4A()
k4a.start()

capture  = k4a.get_capture()
take_picture_print(capture, 'depth_chaire1.png', 'rgb_chair1.png', 'ir_chair1.png')
k4a.stop()
```

## 4. rgb color 가 있는 데이터 vs rgb color 가 있는 데이터 

rgb color 가 있는 데이터는 다음과 같은 코드로 불러온다.

```
xyz = np.loadtxt(불러올 데이터.txt)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# pcd.paint_uniform_color([1,0.706, 0])
o3d.visualization.draw_geometries([pcd])
```

rgb color 가 없는 데이터는 다음과 같은 코드로 불러온다.

```
xyz_color = np.loadtxt(불러올 데이터.txt)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_color[:,:3])
pcd.colors = o3d.utility.Vector3dVector(xyz_color[:,3:])

# pcd.paint_uniform_color([1,0.706, 0])
o3d.visualization.draw_geometries([pcd])
```

pcd, ply 파일형식도 불러오기 가능

```
pcd = o3d.io.read_point_cloud(불러올 데이터.pcd)
pcd = o3d.io.read_point_cloud(불러올 데이터.ply)
```

RGB 데이터와 Depth 데이터가 따로 존재하는 경우 다음과 같은 코드로 불러온다.

```
color_raw = o3d.io.read_image(불러올 데이터.jpg)
depth_raw = o3d.io.read_image(불러올 데이터.png)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

o3d.visualization.draw_geometries([pcd])
```

## 5. 데이터를 효율적으로 불러오는 방법

<img src="https://github.com/bloodmage1/3D_image_processing/blob/main/img/rgb1.png" />

데이터의 크기가 상당히 크기 때문에 데이터의 점의 개수를 선택해서 불러올 수 있다. 그렇게 하면 나중에 전처리 할 때에, 메모리를 효율적으로 사용할 수 있다.

```
mesh = o3d.io.read_triangle_mesh('bunny.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

pcd = mesh.sample_points_uniformly(number_of_points = 500)
o3d.visualization.draw_geometries([pcd])
```

<img src="https://github.com/bloodmage1/3D_image_processing/blob/main/img/rgb_scatter.png" />

위 사진은 데이터에서 500개의 점만 추출하여 화면에 뿌린 것이다.

## 6. 3D IMAGE로 만드는 방법

<img src="https://github.com/bloodmage1/3D_image_processing/blob/main/img/Cup_pictures.png" />
<img src="https://github.com/bloodmage1/3D_image_processing/blob/main/img/Cup_pictures2.png" />
두 사진은 휴지곽 위에 컵을 올려 둔 사진이다(직접 찍음).

1. 먼저 데이터의 배경을 제거
2. 매칭을 위한 파라미터 설정(여기서 distance_ratio, tol이 중요)
3. ransac 함수와 rigid_transform_3D 함수 정의
4. SIFT를 사용한 특징점 매칭 및 3D 변환 계산

<img src="https://github.com/bloodmage1/3D_image_processing/blob/main/img/Cup_merge.png" />


위 데이터는 두 사진에서 비슷한 부분을 찾는 과정이다. 참고로 휴지곽은 컵으로 인해서 생기는 그림자를 제거하기 위해 설치









































































































   

