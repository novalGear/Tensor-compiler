#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func private @forward(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
    %cst = arith.constant dense<[[0.602506042, 0.155216321, -5.401530e-01, -0.503750563, 0.323057234], [-0.423101723, -0.0838975757, -0.458121657, -2.27163601, 0.906597197], [1.62754416, -1.73305011, 1.93884623, 0.0291042831, 1.3649826], [-0.0260805469, 0.411847115, 1.29832327, -0.541960478, 0.102657624], [0.184075668, -0.477209568, -0.884133101, -0.629176437, 0.795449733], [0.727969467, -0.319687724, 0.544110298, -1.16180933, 1.15502715], [1.42709219, -0.416466802, -0.0866447687, -1.00924289, 0.744507074], [-1.46180677, -0.0412837118, 0.0477155335, 1.17928255, 1.85387039], [-0.267419577, 0.694523454, 0.047110226, -0.217776552, -0.609596252], [-0.956114351, 0.542767704, 0.827018141, -2.50129771, -0.65587908]]> : tensor<10x5xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<5xf32>
    %cst_1 = arith.constant dense<2.000000e+00> : tensor<5xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<5xf32>
    %0 = tensor.empty() : tensor<1x5xf32>
    %1 = linalg.matmul ins(%arg0, %cst : tensor<1x10xf32>, tensor<10x5xf32>) outs(%0 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %2 = tensor.empty() : tensor<1x5xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<1x5xf32>, tensor<5xf32>) outs(%2 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %10 = arith.addf %in, %in_3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x5xf32>
    %4 = tensor.empty() : tensor<1x5xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x5xf32>) outs(%4 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_3 = arith.constant 0.000000e+00 : f32
      %10 = arith.maxf %in, %cst_3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x5xf32>
    %6 = tensor.empty() : tensor<1x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_1 : tensor<1x5xf32>, tensor<5xf32>) outs(%6 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %10 = arith.mulf %in, %in_3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x5xf32>
    %8 = tensor.empty() : tensor<1x5xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %cst_2 : tensor<1x5xf32>, tensor<5xf32>) outs(%8 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %10 = arith.addf %in, %in_3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x5xf32>
    return %9 : tensor<1x5xf32>
  }
}
