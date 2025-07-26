Intrinsic matrix (physical units – millimetres):
 tensor([[2.0400e+03, 0.0000e+00, 7.6800e+02],
        [0.0000e+00, 2.0400e+03, 7.6800e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]])
Intrinsic matrix (pixel units):
 tensor([[4.0800e+03, 0.0000e+00, 1.5360e+03],
        [0.0000e+00, 4.0800e+03, 1.5360e+03],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]])
Left   extrinsic (world → detector):
 tensor([[[  1.,   0.,   0.,  50.],
         [  0.,   1.,   0., 850.],
         [  0.,   0.,   1.,   0.],
         [  0.,   0.,   0.,   1.]]])
Right  extrinsic (world → detector):
 tensor([[[  1.,   0.,   0., -50.],
         [  0.,   1.,   0., 850.],
         [  0.,   0.,   1.,   0.],
         [  0.,   0.,   0.,   1.]]])
Left   camera→world  matrix:
 tensor([[[   1.,    0.,    0.,  -50.],
         [   0.,    1.,    0., -850.],
         [   0.,    0.,    1.,   -0.],
         [   0.,    0.,    0.,    1.]]])
Right  camera→world  matrix:
 tensor([[[   1.,    0.,    0.,   50.],
         [   0.,    1.,    0., -850.],
         [   0.,    0.,    1.,   -0.],
         [   0.,    0.,    0.,    1.]]])

Essential matrix  (E):
 tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.7071],
        [ 0.0000,  0.7071,  0.0000]])

Fundamental matrix (F):
 tensor([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.7071],
        [ 0.0000,  0.7071,  0.0000]])

Tensor shape: torch.Size([1, 1, 1536, 1536])
Tensor dtype: torch.float32
Tensor min: 0.0000, max: 24.8577