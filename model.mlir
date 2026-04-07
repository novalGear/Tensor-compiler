#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x10xf32>) -> tensor<1x5xf32>, sym_name = "forward", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1x10xf32>):
    %0 = "arith.constant"() <{value = dense<[[0.602506042, 0.155216321, -5.401530e-01, -0.503750563, 0.323057234], [-0.423101723, -0.0838975757, -0.458121657, -2.27163601, 0.906597197], [1.62754416, -1.73305011, 1.93884623, 0.0291042831, 1.3649826], [-0.0260805469, 0.411847115, 1.29832327, -0.541960478, 0.102657624], [0.184075668, -0.477209568, -0.884133101, -0.629176437, 0.795449733], [0.727969467, -0.319687724, 0.544110298, -1.16180933, 1.15502715], [1.42709219, -0.416466802, -0.0866447687, -1.00924289, 0.744507074], [-1.46180677, -0.0412837118, 0.0477155335, 1.17928255, 1.85387039], [-0.267419577, 0.694523454, 0.047110226, -0.217776552, -0.609596252], [-0.956114351, 0.542767704, 0.827018141, -2.50129771, -0.65587908]]> : tensor<10x5xf32>}> : () -> tensor<10x5xf32>
    %1 = "arith.constant"() <{value = dense<5.000000e-01> : tensor<5xf32>}> : () -> tensor<5xf32>
    %2 = "arith.constant"() <{value = dense<2.000000e+00> : tensor<5xf32>}> : () -> tensor<5xf32>
    %3 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<5xf32>}> : () -> tensor<5xf32>
    %4 = "tensor.empty"() : () -> tensor<1x5xf32>
    %5 = "linalg.matmul"(%arg0, %0, %4) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %14 = "arith.mulf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %15 = "arith.addf"(%arg3, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%15) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<1x10xf32>, tensor<10x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    %6 = "tensor.empty"() : () -> tensor<1x5xf32>
    %7 = "linalg.generic"(%5, %1, %6) <{indexing_maps = [#map3, #map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %14 = "arith.addf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%14) : (f32) -> ()
    }) : (tensor<1x5xf32>, tensor<5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    %8 = "tensor.empty"() : () -> tensor<1x5xf32>
    %9 = "linalg.generic"(%7, %8) <{indexing_maps = [#map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %14 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %15 = "arith.maxf"(%arg1, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%15) : (f32) -> ()
    }) : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    %10 = "tensor.empty"() : () -> tensor<1x5xf32>
    %11 = "linalg.generic"(%9, %2, %10) <{indexing_maps = [#map3, #map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %14 = "arith.mulf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%14) : (f32) -> ()
    }) : (tensor<1x5xf32>, tensor<5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    %12 = "tensor.empty"() : () -> tensor<1x5xf32>
    %13 = "linalg.generic"(%11, %3, %12) <{indexing_maps = [#map3, #map3, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %14 = "arith.addf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%14) : (f32) -> ()
    }) : (tensor<1x5xf32>, tensor<5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    "func.return"(%13) : (tensor<1x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()
