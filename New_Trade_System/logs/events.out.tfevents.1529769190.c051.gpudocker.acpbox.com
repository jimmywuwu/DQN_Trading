       �K"	  �9���Abrain.Event:2CY��;     k��	���9���A"��
d
sPlaceholder*
dtype0*'
_output_shapes
:���������)*
shape:���������)
k
Q_targetPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
m
eval_net/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
`
eval_net/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
eval_net/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+eval_net/random_normal/RandomStandardNormalRandomStandardNormaleval_net/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
eval_net/random_normal/mulMul+eval_net/random_normal/RandomStandardNormaleval_net/random_normal/stddev*
_output_shapes

:
*
T0

eval_net/random_normalAddeval_net/random_normal/muleval_net/random_normal/mean*
T0*
_output_shapes

:

�
eval_net/Variable
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
eval_net/Variable/AssignAssigneval_net/Variableeval_net/random_normal*
use_locking(*
T0*$
_class
loc:@eval_net/Variable*
validate_shape(*
_output_shapes

:

�
eval_net/Variable/readIdentityeval_net/Variable*
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

o
eval_net/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
b
eval_net/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
eval_net/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-eval_net/random_normal_1/RandomStandardNormalRandomStandardNormaleval_net/random_normal_1/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
eval_net/random_normal_1/mulMul-eval_net/random_normal_1/RandomStandardNormaleval_net/random_normal_1/stddev*
T0*
_output_shapes

:

�
eval_net/random_normal_1Addeval_net/random_normal_1/muleval_net/random_normal_1/mean*
T0*
_output_shapes

:

�
eval_net/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
eval_net/Variable_1/AssignAssigneval_net/Variable_1eval_net/random_normal_1*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_1
�
eval_net/Variable_1/readIdentityeval_net/Variable_1*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

[
eval_net/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:


eval_net/Variable_2
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
eval_net/Variable_2/AssignAssigneval_net/Variable_2eval_net/Const*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2*
validate_shape(*
_output_shapes
:

�
eval_net/Variable_2/readIdentityeval_net/Variable_2*
_output_shapes
:
*
T0*&
_class
loc:@eval_net/Variable_2
]
eval_net/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:

eval_net/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
eval_net/Variable_3/AssignAssigneval_net/Variable_3eval_net/Const_1*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_3*
validate_shape(*
_output_shapes
:
�
eval_net/Variable_3/readIdentityeval_net/Variable_3*
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:
k
eval_net/Reshape/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:
z
eval_net/ReshapeReshapeseval_net/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:���������
i
eval_net/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
eval_net/Reshape_1Reshapeeval_net/Reshapeeval_net/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
eval_net/MatMulMatMuleval_net/Reshape_1eval_net/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
p
eval_net/addAddeval_net/MatMuleval_net/Variable_2/read*
T0*'
_output_shapes
:���������

m
eval_net/Reshape_2/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
eval_net/Reshape_2Reshapeeval_net/addeval_net/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������

o
%eval_net/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
q
'eval_net/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
m
+eval_net/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
&eval_net/BasicLSTMCellZeroState/concatConcatV2%eval_net/BasicLSTMCellZeroState/Const'eval_net/BasicLSTMCellZeroState/Const_1+eval_net/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
p
+eval_net/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%eval_net/BasicLSTMCellZeroState/zerosFill&eval_net/BasicLSTMCellZeroState/concat+eval_net/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:
*
T0
q
'eval_net/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
o
-eval_net/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
(eval_net/BasicLSTMCellZeroState/concat_1ConcatV2'eval_net/BasicLSTMCellZeroState/Const_4'eval_net/BasicLSTMCellZeroState/Const_5-eval_net/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
r
-eval_net/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'eval_net/BasicLSTMCellZeroState/zeros_1Fill(eval_net/BasicLSTMCellZeroState/concat_1-eval_net/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

q
'eval_net/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
O
eval_net/RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
eval_net/range/startConst*
dtype0*
_output_shapes
: *
value	B :
V
eval_net/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
eval_net/rangeRangeeval_net/range/starteval_net/Rankeval_net/range/delta*

Tidx0*
_output_shapes
:
i
eval_net/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
V
eval_net/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/concatConcatV2eval_net/concat/values_0eval_net/rangeeval_net/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
eval_net/transpose	Transposeeval_net/Reshape_2eval_net/concat*
T0*+
_output_shapes
:���������
*
Tperm0
d
eval_net/rnn/ShapeShapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
j
 eval_net/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
l
"eval_net/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"eval_net/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/rnn/strided_sliceStridedSliceeval_net/rnn/Shape eval_net/rnn/strided_slice/stack"eval_net/rnn/strided_slice/stack_1"eval_net/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
eval_net/rnn/Shape_1Shapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
l
"eval_net/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$eval_net/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
n
$eval_net/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
eval_net/rnn/strided_slice_1StridedSliceeval_net/rnn/Shape_1"eval_net/rnn/strided_slice_1/stack$eval_net/rnn/strided_slice_1/stack_1$eval_net/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
eval_net/rnn/Shape_2Shapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
l
"eval_net/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
n
$eval_net/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
n
$eval_net/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/rnn/strided_slice_2StridedSliceeval_net/rnn/Shape_2"eval_net/rnn/strided_slice_2/stack$eval_net/rnn/strided_slice_2/stack_1$eval_net/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
]
eval_net/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/ExpandDims
ExpandDimseval_net/rnn/strided_slice_2eval_net/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
\
eval_net/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Z
eval_net/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/concatConcatV2eval_net/rnn/ExpandDimseval_net/rnn/Consteval_net/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
]
eval_net/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
eval_net/rnn/zerosFilleval_net/rnn/concateval_net/rnn/zeros/Const*
T0*'
_output_shapes
:���������

S
eval_net/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/TensorArrayTensorArrayV3eval_net/rnn/strided_slice_1*8
tensor_array_name#!eval_net/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
�
eval_net/rnn/TensorArray_1TensorArrayV3eval_net/rnn/strided_slice_1*7
tensor_array_name" eval_net/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
w
%eval_net/rnn/TensorArrayUnstack/ShapeShapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
}
3eval_net/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
-eval_net/rnn/TensorArrayUnstack/strided_sliceStridedSlice%eval_net/rnn/TensorArrayUnstack/Shape3eval_net/rnn/TensorArrayUnstack/strided_slice/stack5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_15eval_net/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
+eval_net/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
m
+eval_net/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
%eval_net/rnn/TensorArrayUnstack/rangeRange+eval_net/rnn/TensorArrayUnstack/range/start-eval_net/rnn/TensorArrayUnstack/strided_slice+eval_net/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Geval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3eval_net/rnn/TensorArray_1%eval_net/rnn/TensorArrayUnstack/rangeeval_net/transposeeval_net/rnn/TensorArray_1:1*
_output_shapes
: *
T0*%
_class
loc:@eval_net/transpose
�
eval_net/rnn/while/EnterEntereval_net/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
eval_net/rnn/while/Enter_1Entereval_net/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
eval_net/rnn/while/Enter_2Enter%eval_net/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant( 
�
eval_net/rnn/while/Enter_3Enter'eval_net/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*0

frame_name" eval_net/rnn/while/while_context
�
eval_net/rnn/while/MergeMergeeval_net/rnn/while/Enter eval_net/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
eval_net/rnn/while/Merge_1Mergeeval_net/rnn/while/Enter_1"eval_net/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
eval_net/rnn/while/Merge_2Mergeeval_net/rnn/while/Enter_2"eval_net/rnn/while/NextIteration_2*
N* 
_output_shapes
:
: *
T0
�
eval_net/rnn/while/Merge_3Mergeeval_net/rnn/while/Enter_3"eval_net/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
eval_net/rnn/while/Less/EnterEntereval_net/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
y
eval_net/rnn/while/LessLesseval_net/rnn/while/Mergeeval_net/rnn/while/Less/Enter*
T0*
_output_shapes
: 
X
eval_net/rnn/while/LoopCondLoopCondeval_net/rnn/while/Less*
_output_shapes
: 
�
eval_net/rnn/while/SwitchSwitcheval_net/rnn/while/Mergeeval_net/rnn/while/LoopCond*
T0*+
_class!
loc:@eval_net/rnn/while/Merge*
_output_shapes
: : 
�
eval_net/rnn/while/Switch_1Switcheval_net/rnn/while/Merge_1eval_net/rnn/while/LoopCond*
_output_shapes
: : *
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_1
�
eval_net/rnn/while/Switch_2Switcheval_net/rnn/while/Merge_2eval_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_2*(
_output_shapes
:
:

�
eval_net/rnn/while/Switch_3Switcheval_net/rnn/while/Merge_3eval_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_3*(
_output_shapes
:
:

e
eval_net/rnn/while/IdentityIdentityeval_net/rnn/while/Switch:1*
T0*
_output_shapes
: 
i
eval_net/rnn/while/Identity_1Identityeval_net/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
q
eval_net/rnn/while/Identity_2Identityeval_net/rnn/while/Switch_2:1*
_output_shapes

:
*
T0
q
eval_net/rnn/while/Identity_3Identityeval_net/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
*eval_net/rnn/while/TensorArrayReadV3/EnterEntereval_net/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
,eval_net/rnn/while/TensorArrayReadV3/Enter_1EnterGeval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
$eval_net/rnn/while/TensorArrayReadV3TensorArrayReadV3*eval_net/rnn/while/TensorArrayReadV3/Entereval_net/rnn/while/Identity,eval_net/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
Deval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB
 *�衾*
dtype0*
_output_shapes
: 
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Leval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformDeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed 
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSubBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulLeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
_output_shapes

:(*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel
�
>eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAddBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
#eval_net/rnn/basic_lstm_cell/kernel
VariableV2*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
*eval_net/rnn/basic_lstm_cell/kernel/AssignAssign#eval_net/rnn/basic_lstm_cell/kernel>eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel
�
(eval_net/rnn/basic_lstm_cell/kernel/readIdentity#eval_net/rnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
3eval_net/rnn/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:(*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*    
�
!eval_net/rnn/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
(eval_net/rnn/basic_lstm_cell/bias/AssignAssign!eval_net/rnn/basic_lstm_cell/bias3eval_net/rnn/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias
z
&eval_net/rnn/basic_lstm_cell/bias/readIdentity!eval_net/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
�
2eval_net/rnn/while/rnn/basic_lstm_cell/concat/axisConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
-eval_net/rnn/while/rnn/basic_lstm_cell/concatConcatV2$eval_net/rnn/while/TensorArrayReadV3eval_net/rnn/while/Identity_32eval_net/rnn/while/rnn/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
3eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/EnterEnter(eval_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*0

frame_name" eval_net/rnn/while/while_context
�
-eval_net/rnn/while/rnn/basic_lstm_cell/MatMulMatMul-eval_net/rnn/while/rnn/basic_lstm_cell/concat3eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter*
_output_shapes

:(*
transpose_a( *
transpose_b( *
T0
�
4eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/EnterEnter&eval_net/rnn/basic_lstm_cell/bias/read*
parallel_iterations *
_output_shapes
:(*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
�
.eval_net/rnn/while/rnn/basic_lstm_cell/BiasAddBiasAdd-eval_net/rnn/while/rnn/basic_lstm_cell/MatMul4eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
,eval_net/rnn/while/rnn/basic_lstm_cell/ConstConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
6eval_net/rnn/while/rnn/basic_lstm_cell/split/split_dimConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
,eval_net/rnn/while/rnn/basic_lstm_cell/splitSplit6eval_net/rnn/while/rnn/basic_lstm_cell/split/split_dim.eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
,eval_net/rnn/while/rnn/basic_lstm_cell/add/yConst^eval_net/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*eval_net/rnn/while/rnn/basic_lstm_cell/addAdd.eval_net/rnn/while/rnn/basic_lstm_cell/split:2,eval_net/rnn/while/rnn/basic_lstm_cell/add/y*
T0*
_output_shapes

:

�
.eval_net/rnn/while/rnn/basic_lstm_cell/SigmoidSigmoid*eval_net/rnn/while/rnn/basic_lstm_cell/add*
_output_shapes

:
*
T0
�
*eval_net/rnn/while/rnn/basic_lstm_cell/mulMuleval_net/rnn/while/Identity_2.eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1Sigmoid,eval_net/rnn/while/rnn/basic_lstm_cell/split*
_output_shapes

:
*
T0
�
+eval_net/rnn/while/rnn/basic_lstm_cell/TanhTanh.eval_net/rnn/while/rnn/basic_lstm_cell/split:1*
_output_shapes

:
*
T0
�
,eval_net/rnn/while/rnn/basic_lstm_cell/mul_1Mul0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1+eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*
_output_shapes

:
*
T0
�
,eval_net/rnn/while/rnn/basic_lstm_cell/add_1Add*eval_net/rnn/while/rnn/basic_lstm_cell/mul,eval_net/rnn/while/rnn/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1Tanh,eval_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2Sigmoid.eval_net/rnn/while/rnn/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2Mul-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_10eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
<eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntereval_net/rnn/TensorArray*
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context*
T0*?
_class5
31loc:@eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
parallel_iterations *
is_constant(
�
6eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3<eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/rnn/while/Identity,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2eval_net/rnn/while/Identity_1*
T0*?
_class5
31loc:@eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
_output_shapes
: 
x
eval_net/rnn/while/add/yConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
eval_net/rnn/while/addAddeval_net/rnn/while/Identityeval_net/rnn/while/add/y*
_output_shapes
: *
T0
j
 eval_net/rnn/while/NextIterationNextIterationeval_net/rnn/while/add*
T0*
_output_shapes
: 
�
"eval_net/rnn/while/NextIteration_1NextIteration6eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
"eval_net/rnn/while/NextIteration_2NextIteration,eval_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
"eval_net/rnn/while/NextIteration_3NextIteration,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
T0*
_output_shapes

:

[
eval_net/rnn/while/ExitExiteval_net/rnn/while/Switch*
T0*
_output_shapes
: 
_
eval_net/rnn/while/Exit_1Exiteval_net/rnn/while/Switch_1*
_output_shapes
: *
T0
g
eval_net/rnn/while/Exit_2Exiteval_net/rnn/while/Switch_2*
T0*
_output_shapes

:

g
eval_net/rnn/while/Exit_3Exiteval_net/rnn/while/Switch_3*
T0*
_output_shapes

:

�
/eval_net/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3eval_net/rnn/TensorArrayeval_net/rnn/while/Exit_1*+
_class!
loc:@eval_net/rnn/TensorArray*
_output_shapes
: 
�
)eval_net/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *+
_class!
loc:@eval_net/rnn/TensorArray
�
)eval_net/rnn/TensorArrayStack/range/deltaConst*
value	B :*+
_class!
loc:@eval_net/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
#eval_net/rnn/TensorArrayStack/rangeRange)eval_net/rnn/TensorArrayStack/range/start/eval_net/rnn/TensorArrayStack/TensorArraySizeV3)eval_net/rnn/TensorArrayStack/range/delta*+
_class!
loc:@eval_net/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
1eval_net/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3eval_net/rnn/TensorArray#eval_net/rnn/TensorArrayStack/rangeeval_net/rnn/while/Exit_1*
element_shape
:
*+
_class!
loc:@eval_net/rnn/TensorArray*
dtype0*"
_output_shapes
:

^
eval_net/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
S
eval_net/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Z
eval_net/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Z
eval_net/rnn/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
eval_net/rnn/rangeRangeeval_net/rnn/range/starteval_net/rnn/Rankeval_net/rnn/range/delta*
_output_shapes
:*

Tidx0
o
eval_net/rnn/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
\
eval_net/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/concat_1ConcatV2eval_net/rnn/concat_1/values_0eval_net/rnn/rangeeval_net/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
eval_net/rnn/transpose	Transpose1eval_net/rnn/TensorArrayStack/TensorArrayGatherV3eval_net/rnn/concat_1*"
_output_shapes
:
*
Tperm0*
T0
�
eval_net/MatMul_1MatMuleval_net/rnn/while/Exit_3eval_net/Variable_1/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
k
eval_net/add_1Addeval_net/MatMul_1eval_net/Variable_3/read*
_output_shapes

:*
T0
w
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/add_1*
T0*'
_output_shapes
:���������
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	loss/MeanMeanloss/SquaredDifference
loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
_output_shapes
: *
T0
Y
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
train/gradients/SwitchSwitchtrain/gradients/Mergeeval_net/rnn/while/LoopCond*
T0*
_output_shapes
: : 
u
train/gradients/Add/yConst^eval_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/AddR^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPushV2T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2P^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPushV2R^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPushV2R^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPushV2T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2X^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2V^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2X^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1j^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2*
T0*
_output_shapes
: 
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
T0*
_output_shapes
: 
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
_output_shapes
: : *
T0
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
train/gradients/NextIteration_1NextIterationtrain/gradients/SubM^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/b_sync*
_output_shapes
: *
T0
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
T0*
_output_shapes
: 
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/loss/Mean_grad/ShapeShapeloss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/loss/SquaredDifference_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/add_1'^train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:
�
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
�
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg*
_output_shapes

:
z
)train/gradients/eval_net/add_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
u
+train/gradients/eval_net/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
9train/gradients/eval_net/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs)train/gradients/eval_net/add_1_grad/Shape+train/gradients/eval_net/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
'train/gradients/eval_net/add_1_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_19train/gradients/eval_net/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
+train/gradients/eval_net/add_1_grad/ReshapeReshape'train/gradients/eval_net/add_1_grad/Sum)train/gradients/eval_net/add_1_grad/Shape*
_output_shapes

:*
T0*
Tshape0
�
)train/gradients/eval_net/add_1_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1;train/gradients/eval_net/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
-train/gradients/eval_net/add_1_grad/Reshape_1Reshape)train/gradients/eval_net/add_1_grad/Sum_1+train/gradients/eval_net/add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
4train/gradients/eval_net/add_1_grad/tuple/group_depsNoOp,^train/gradients/eval_net/add_1_grad/Reshape.^train/gradients/eval_net/add_1_grad/Reshape_1
�
<train/gradients/eval_net/add_1_grad/tuple/control_dependencyIdentity+train/gradients/eval_net/add_1_grad/Reshape5^train/gradients/eval_net/add_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/add_1_grad/Reshape*
_output_shapes

:
�
>train/gradients/eval_net/add_1_grad/tuple/control_dependency_1Identity-train/gradients/eval_net/add_1_grad/Reshape_15^train/gradients/eval_net/add_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/eval_net/add_1_grad/Reshape_1*
_output_shapes
:
�
-train/gradients/eval_net/MatMul_1_grad/MatMulMatMul<train/gradients/eval_net/add_1_grad/tuple/control_dependencyeval_net/Variable_1/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
/train/gradients/eval_net/MatMul_1_grad/MatMul_1MatMuleval_net/rnn/while/Exit_3<train/gradients/eval_net/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
7train/gradients/eval_net/MatMul_1_grad/tuple/group_depsNoOp.^train/gradients/eval_net/MatMul_1_grad/MatMul0^train/gradients/eval_net/MatMul_1_grad/MatMul_1
�
?train/gradients/eval_net/MatMul_1_grad/tuple/control_dependencyIdentity-train/gradients/eval_net/MatMul_1_grad/MatMul8^train/gradients/eval_net/MatMul_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/eval_net/MatMul_1_grad/MatMul*
_output_shapes

:

�
Atrain/gradients/eval_net/MatMul_1_grad/tuple/control_dependency_1Identity/train/gradients/eval_net/MatMul_1_grad/MatMul_18^train/gradients/eval_net/MatMul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/eval_net/MatMul_1_grad/MatMul_1*
_output_shapes

:

Z
train/gradients/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
train/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
5train/gradients/eval_net/rnn/while/Exit_3_grad/b_exitEnter?train/gradients/eval_net/MatMul_1_grad/tuple/control_dependency*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
5train/gradients/eval_net/rnn/while/Exit_1_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
5train/gradients/eval_net/rnn/while/Exit_2_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
9train/gradients/eval_net/rnn/while/Switch_3_grad/b_switchMerge5train/gradients/eval_net/rnn/while/Exit_3_grad/b_exit@train/gradients/eval_net/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
9train/gradients/eval_net/rnn/while/Switch_2_grad/b_switchMerge5train/gradients/eval_net/rnn/while/Exit_2_grad/b_exit@train/gradients/eval_net/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
6train/gradients/eval_net/rnn/while/Merge_3_grad/SwitchSwitch9train/gradients/eval_net/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

�
@train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_depsNoOp7^train/gradients/eval_net/rnn/while/Merge_3_grad/Switch
�
Htrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity6train/gradients/eval_net/rnn/while/Merge_3_grad/SwitchA^train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Jtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity8train/gradients/eval_net/rnn/while/Merge_3_grad/Switch:1A^train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
6train/gradients/eval_net/rnn/while/Merge_2_grad/SwitchSwitch9train/gradients/eval_net/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*(
_output_shapes
:
:

�
@train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_depsNoOp7^train/gradients/eval_net/rnn/while/Merge_2_grad/Switch
�
Htrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity6train/gradients/eval_net/rnn/while/Merge_2_grad/SwitchA^train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch
�
Jtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity8train/gradients/eval_net/rnn/while/Merge_2_grad/Switch:1A^train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
4train/gradients/eval_net/rnn/while/Enter_3_grad/ExitExitHtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
4train/gradients/eval_net/rnn/while/Enter_2_grad/ExitExitHtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
	elem_type0*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/Enter0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/b_syncControlTriggerQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2O^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2W^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2U^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2W^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1i^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mulMulJtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/SumSumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mulWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1*
dtype0*
_output_shapes
: 
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/Enter-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1MulRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Jtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Sum_1SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
train/gradients/AddNAddNJtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency_1Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*
N*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/SumSumtrain/gradients/AddNWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Sum_1Sumtrain/gradients/AddNYtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:

�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ShapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*A
_class7
53loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
dtype0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_accStackV2Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc/max_size*A
_class7
53loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/EnterEnterItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/Enter.eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mulMulZtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependencyNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/SumSumCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mulUtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeReshapeCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*0
_class&
$"loc:@eval_net/rnn/while/Identity_2*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*0
_class&
$"loc:@eval_net/rnn/while/Identity_2
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/Entereval_net/rnn/while/Identity_2^train/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1MulPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Sum_1SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1ReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Sum_1Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_depsNoOpH^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependencyIdentityGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:

�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
valueB :
���������*>
_class4
20loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*>
_class4
20loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/Enter+eval_net/rnn/while/rnn/basic_lstm_cell/Tanh^train/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mulMul\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:
*
T0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/SumSumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mulWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1*
dtype0*
_output_shapes
: 
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/Enter0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1MulRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Sum_1SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:

�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_grad/TanhGradTanhGradPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
@train/gradients/eval_net/rnn/while/Switch_2_grad_1/NextIterationNextIterationXtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ShapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/SumSumOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradUtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeReshapeCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Sum_1SumOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1ReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Sum_1Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_depsNoOpH^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependencyIdentityGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape*
_output_shapes

:

�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatConcatV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_grad/TanhGradXtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependencyQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes

:(*

Tidx0
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpI^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatP^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatU^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradU^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter(eval_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulMatMul\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul/Enter*
_output_shapes

:*
transpose_a( *
transpose_b(*
T0
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
valueB :
���������*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/concat*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/Enter-eval_net/rnn/while/rnn/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1MatMulVtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a(
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
�
[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulT^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:(*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes
:(*
valueB(*    
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/AddAddRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/mod/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
Ftrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/modFloorModLtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/mod/ConstGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeShape$eval_net/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_sizeConst*
valueB :
���������*7
_class-
+)loc:@eval_net/rnn/while/TensorArrayReadV3*
dtype0*
_output_shapes
: 
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_size*
	elem_type0*7
_class-
+)loc:@eval_net/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/EnterEnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter$eval_net/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*'
_output_shapes
:���������
*
	elem_type0
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_sizeConst*
valueB :
���������*0
_class&
$"loc:@eval_net/rnn/while/Identity_3*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_size*
	elem_type0*0
_class&
$"loc:@eval_net/rnn/while/Identity_3*

stack_name *
_output_shapes
:
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1eval_net/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeNShapeNTtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetFtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/modItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeNKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceSlice[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffsetItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:������������������
�
Jtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1Slice[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffset:1Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:������������������
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_depsNoOpI^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceK^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1
�
[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceT^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityJtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*]
_classS
QOloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergePtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/AddAddQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Switch:1]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:(*
T0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationLtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
atrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntereval_net/rnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterGeval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
[train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3atrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*
sourcetrain/gradients*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
�
Wtrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1\^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
ltrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc/max_sizeConst*
valueB :
���������*.
_class$
" loc:@eval_net/rnn/while/Identity*
dtype0*
_output_shapes
: 
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStackV2ltrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc/max_size*.
_class$
" loc:@eval_net/rnn/while/Identity*

stack_name *
_output_shapes
:*
	elem_type0
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/EnterEnterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
itrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2StackPushV2ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/rnn/while/Identity^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
ntrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2/EnterEnterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
htrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2
StackPopV2ntrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
]train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3[train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3htrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependencyWtrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
�
Gtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterGtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Otrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
_output_shapes
: : *
T0
�
Htrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Etrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddJtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1]train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
Otrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationEtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitHtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
@train/gradients/eval_net/rnn/while/Switch_3_grad_1/NextIterationNextIteration]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
~train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3eval_net/rnn/TensorArray_1Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *
sourcetrain/gradients*-
_class#
!loc:@eval_net/rnn/TensorArray_1
�
ztrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*-
_class#
!loc:@eval_net/rnn/TensorArray_1*
_output_shapes
: 
�
ptrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3~train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3%eval_net/rnn/TensorArrayUnstack/rangeztrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:*
element_shape:
�
mtrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3q^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
utrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityptrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3n^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*�
_classy
wuloc:@train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*+
_output_shapes
:���������

�
wtrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3n^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
9train/gradients/eval_net/transpose_grad/InvertPermutationInvertPermutationeval_net/concat*
T0*
_output_shapes
:
�
1train/gradients/eval_net/transpose_grad/transpose	Transposeutrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency9train/gradients/eval_net/transpose_grad/InvertPermutation*
T0*+
_output_shapes
:���������
*
Tperm0
y
-train/gradients/eval_net/Reshape_2_grad/ShapeShapeeval_net/add*
_output_shapes
:*
T0*
out_type0
�
/train/gradients/eval_net/Reshape_2_grad/ReshapeReshape1train/gradients/eval_net/transpose_grad/transpose-train/gradients/eval_net/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

v
'train/gradients/eval_net/add_grad/ShapeShapeeval_net/MatMul*
T0*
out_type0*
_output_shapes
:
s
)train/gradients/eval_net/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
7train/gradients/eval_net/add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/eval_net/add_grad/Shape)train/gradients/eval_net/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%train/gradients/eval_net/add_grad/SumSum/train/gradients/eval_net/Reshape_2_grad/Reshape7train/gradients/eval_net/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
)train/gradients/eval_net/add_grad/ReshapeReshape%train/gradients/eval_net/add_grad/Sum'train/gradients/eval_net/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
'train/gradients/eval_net/add_grad/Sum_1Sum/train/gradients/eval_net/Reshape_2_grad/Reshape9train/gradients/eval_net/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
+train/gradients/eval_net/add_grad/Reshape_1Reshape'train/gradients/eval_net/add_grad/Sum_1)train/gradients/eval_net/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
2train/gradients/eval_net/add_grad/tuple/group_depsNoOp*^train/gradients/eval_net/add_grad/Reshape,^train/gradients/eval_net/add_grad/Reshape_1
�
:train/gradients/eval_net/add_grad/tuple/control_dependencyIdentity)train/gradients/eval_net/add_grad/Reshape3^train/gradients/eval_net/add_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*<
_class2
0.loc:@train/gradients/eval_net/add_grad/Reshape
�
<train/gradients/eval_net/add_grad/tuple/control_dependency_1Identity+train/gradients/eval_net/add_grad/Reshape_13^train/gradients/eval_net/add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/add_grad/Reshape_1*
_output_shapes
:

�
+train/gradients/eval_net/MatMul_grad/MatMulMatMul:train/gradients/eval_net/add_grad/tuple/control_dependencyeval_net/Variable/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
-train/gradients/eval_net/MatMul_grad/MatMul_1MatMuleval_net/Reshape_1:train/gradients/eval_net/add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
�
5train/gradients/eval_net/MatMul_grad/tuple/group_depsNoOp,^train/gradients/eval_net/MatMul_grad/MatMul.^train/gradients/eval_net/MatMul_grad/MatMul_1
�
=train/gradients/eval_net/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/eval_net/MatMul_grad/MatMul6^train/gradients/eval_net/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/MatMul_grad/MatMul*'
_output_shapes
:���������
�
?train/gradients/eval_net/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/eval_net/MatMul_grad/MatMul_16^train/gradients/eval_net/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/eval_net/MatMul_grad/MatMul_1*
_output_shapes

:

�
0train/eval_net/Variable/RMSProp/Initializer/onesConst*$
_class
loc:@eval_net/Variable*
valueB
*  �?*
dtype0*
_output_shapes

:

�
train/eval_net/Variable/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *$
_class
loc:@eval_net/Variable*
	container *
shape
:

�
&train/eval_net/Variable/RMSProp/AssignAssigntrain/eval_net/Variable/RMSProp0train/eval_net/Variable/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*$
_class
loc:@eval_net/Variable
�
$train/eval_net/Variable/RMSProp/readIdentitytrain/eval_net/Variable/RMSProp*
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

�
3train/eval_net/Variable/RMSProp_1/Initializer/zerosConst*$
_class
loc:@eval_net/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
!train/eval_net/Variable/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *$
_class
loc:@eval_net/Variable*
	container *
shape
:

�
(train/eval_net/Variable/RMSProp_1/AssignAssign!train/eval_net/Variable/RMSProp_13train/eval_net/Variable/RMSProp_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@eval_net/Variable*
validate_shape(*
_output_shapes

:

�
&train/eval_net/Variable/RMSProp_1/readIdentity!train/eval_net/Variable/RMSProp_1*
_output_shapes

:
*
T0*$
_class
loc:@eval_net/Variable
�
2train/eval_net/Variable_1/RMSProp/Initializer/onesConst*&
_class
loc:@eval_net/Variable_1*
valueB
*  �?*
dtype0*
_output_shapes

:

�
!train/eval_net/Variable_1/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *&
_class
loc:@eval_net/Variable_1*
	container *
shape
:

�
(train/eval_net/Variable_1/RMSProp/AssignAssign!train/eval_net/Variable_1/RMSProp2train/eval_net/Variable_1/RMSProp/Initializer/ones*
T0*&
_class
loc:@eval_net/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
&train/eval_net/Variable_1/RMSProp/readIdentity!train/eval_net/Variable_1/RMSProp*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

�
5train/eval_net/Variable_1/RMSProp_1/Initializer/zerosConst*&
_class
loc:@eval_net/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
#train/eval_net/Variable_1/RMSProp_1
VariableV2*&
_class
loc:@eval_net/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
*train/eval_net/Variable_1/RMSProp_1/AssignAssign#train/eval_net/Variable_1/RMSProp_15train/eval_net/Variable_1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_1
�
(train/eval_net/Variable_1/RMSProp_1/readIdentity#train/eval_net/Variable_1/RMSProp_1*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

�
2train/eval_net/Variable_2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes
:
*&
_class
loc:@eval_net/Variable_2*
valueB
*  �?
�
!train/eval_net/Variable_2/RMSProp
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *&
_class
loc:@eval_net/Variable_2*
	container 
�
(train/eval_net/Variable_2/RMSProp/AssignAssign!train/eval_net/Variable_2/RMSProp2train/eval_net/Variable_2/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2
�
&train/eval_net/Variable_2/RMSProp/readIdentity!train/eval_net/Variable_2/RMSProp*
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

�
5train/eval_net/Variable_2/RMSProp_1/Initializer/zerosConst*&
_class
loc:@eval_net/Variable_2*
valueB
*    *
dtype0*
_output_shapes
:

�
#train/eval_net/Variable_2/RMSProp_1
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *&
_class
loc:@eval_net/Variable_2*
	container *
shape:

�
*train/eval_net/Variable_2/RMSProp_1/AssignAssign#train/eval_net/Variable_2/RMSProp_15train/eval_net/Variable_2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2*
validate_shape(*
_output_shapes
:

�
(train/eval_net/Variable_2/RMSProp_1/readIdentity#train/eval_net/Variable_2/RMSProp_1*
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

�
2train/eval_net/Variable_3/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes
:*&
_class
loc:@eval_net/Variable_3*
valueB*  �?
�
!train/eval_net/Variable_3/RMSProp
VariableV2*&
_class
loc:@eval_net/Variable_3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
(train/eval_net/Variable_3/RMSProp/AssignAssign!train/eval_net/Variable_3/RMSProp2train/eval_net/Variable_3/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_3
�
&train/eval_net/Variable_3/RMSProp/readIdentity!train/eval_net/Variable_3/RMSProp*
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:
�
5train/eval_net/Variable_3/RMSProp_1/Initializer/zerosConst*&
_class
loc:@eval_net/Variable_3*
valueB*    *
dtype0*
_output_shapes
:
�
#train/eval_net/Variable_3/RMSProp_1
VariableV2*
shared_name *&
_class
loc:@eval_net/Variable_3*
	container *
shape:*
dtype0*
_output_shapes
:
�
*train/eval_net/Variable_3/RMSProp_1/AssignAssign#train/eval_net/Variable_3/RMSProp_15train/eval_net/Variable_3/RMSProp_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_3*
validate_shape(*
_output_shapes
:
�
(train/eval_net/Variable_3/RMSProp_1/readIdentity#train/eval_net/Variable_3/RMSProp_1*
_output_shapes
:*
T0*&
_class
loc:@eval_net/Variable_3
�
Btrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Initializer/onesConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB(*  �?*
dtype0*
_output_shapes

:(
�
1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container 
�
8train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/AssignAssign1train/eval_net/rnn/basic_lstm_cell/kernel/RMSPropBtrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
6train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/readIdentity1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
Etrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:(*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB(*    
�
3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container 
�
:train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/AssignAssign3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1Etrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
8train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/readIdentity3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
@train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Initializer/onesConst*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*  �?*
dtype0*
_output_shapes
:(
�
/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp
VariableV2*
shared_name *4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
6train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/AssignAssign/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp@train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias
�
4train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/readIdentity/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
Ctrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*    
�
1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
8train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/AssignAssign1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1Ctrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias
�
6train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/readIdentity1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
_output_shapes
:(
`
train/RMSProp/learning_rateConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
X
train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
[
train/RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
3train/RMSProp/update_eval_net/Variable/ApplyRMSPropApplyRMSPropeval_net/Variabletrain/eval_net/Variable/RMSProp!train/eval_net/Variable/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*$
_class
loc:@eval_net/Variable
�
5train/RMSProp/update_eval_net/Variable_1/ApplyRMSPropApplyRMSPropeval_net/Variable_1!train/eval_net/Variable_1/RMSProp#train/eval_net/Variable_1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonAtrain/gradients/eval_net/MatMul_1_grad/tuple/control_dependency_1*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:
*
use_locking( 
�
5train/RMSProp/update_eval_net/Variable_2/ApplyRMSPropApplyRMSPropeval_net/Variable_2!train/eval_net/Variable_2/RMSProp#train/eval_net/Variable_2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon<train/gradients/eval_net/add_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0*&
_class
loc:@eval_net/Variable_2
�
5train/RMSProp/update_eval_net/Variable_3/ApplyRMSPropApplyRMSPropeval_net/Variable_3!train/eval_net/Variable_3/RMSProp#train/eval_net/Variable_3/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon>train/gradients/eval_net/add_1_grad/tuple/control_dependency_1*
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:*
use_locking( 
�
Etrain/RMSProp/update_eval_net/rnn/basic_lstm_cell/kernel/ApplyRMSPropApplyRMSProp#eval_net/rnn/basic_lstm_cell/kernel1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ctrain/RMSProp/update_eval_net/rnn/basic_lstm_cell/bias/ApplyRMSPropApplyRMSProp!eval_net/rnn/basic_lstm_cell/bias/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
_output_shapes
:(*
use_locking( 
�
train/RMSPropNoOp4^train/RMSProp/update_eval_net/Variable/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_1/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_2/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_3/ApplyRMSPropF^train/RMSProp/update_eval_net/rnn/basic_lstm_cell/kernel/ApplyRMSPropD^train/RMSProp/update_eval_net/rnn/basic_lstm_cell/bias/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:���������)*
shape:���������)
o
target_net/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
b
target_net/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
target_net/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-target_net/random_normal/RandomStandardNormalRandomStandardNormaltarget_net/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:
*
seed2 
�
target_net/random_normal/mulMul-target_net/random_normal/RandomStandardNormaltarget_net/random_normal/stddev*
T0*
_output_shapes

:

�
target_net/random_normalAddtarget_net/random_normal/multarget_net/random_normal/mean*
T0*
_output_shapes

:

�
target_net/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
target_net/Variable/AssignAssigntarget_net/Variabletarget_net/random_normal*
use_locking(*
T0*&
_class
loc:@target_net/Variable*
validate_shape(*
_output_shapes

:

�
target_net/Variable/readIdentitytarget_net/Variable*
T0*&
_class
loc:@target_net/Variable*
_output_shapes

:

q
 target_net/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
d
target_net/random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!target_net/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
/target_net/random_normal_1/RandomStandardNormalRandomStandardNormal target_net/random_normal_1/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
target_net/random_normal_1/mulMul/target_net/random_normal_1/RandomStandardNormal!target_net/random_normal_1/stddev*
T0*
_output_shapes

:

�
target_net/random_normal_1Addtarget_net/random_normal_1/multarget_net/random_normal_1/mean*
T0*
_output_shapes

:

�
target_net/Variable_1
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
target_net/Variable_1/AssignAssigntarget_net/Variable_1target_net/random_normal_1*
T0*(
_class
loc:@target_net/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
target_net/Variable_1/readIdentitytarget_net/Variable_1*
_output_shapes

:
*
T0*(
_class
loc:@target_net/Variable_1
]
target_net/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*���=
�
target_net/Variable_2
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
target_net/Variable_2/AssignAssigntarget_net/Variable_2target_net/Const*
use_locking(*
T0*(
_class
loc:@target_net/Variable_2*
validate_shape(*
_output_shapes
:

�
target_net/Variable_2/readIdentitytarget_net/Variable_2*
T0*(
_class
loc:@target_net/Variable_2*
_output_shapes
:

_
target_net/Const_1Const*
dtype0*
_output_shapes
:*
valueB*���=
�
target_net/Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
target_net/Variable_3/AssignAssigntarget_net/Variable_3target_net/Const_1*
use_locking(*
T0*(
_class
loc:@target_net/Variable_3*
validate_shape(*
_output_shapes
:
�
target_net/Variable_3/readIdentitytarget_net/Variable_3*
T0*(
_class
loc:@target_net/Variable_3*
_output_shapes
:
m
target_net/Reshape/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      

target_net/ReshapeReshapes_target_net/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:���������
k
target_net/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
target_net/Reshape_1Reshapetarget_net/Reshapetarget_net/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
target_net/MatMulMatMultarget_net/Reshape_1target_net/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
v
target_net/addAddtarget_net/MatMultarget_net/Variable_2/read*
T0*'
_output_shapes
:���������

o
target_net/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����   
   
�
target_net/Reshape_2Reshapetarget_net/addtarget_net/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������

q
'target_net/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
s
)target_net/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
o
-target_net/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
(target_net/BasicLSTMCellZeroState/concatConcatV2'target_net/BasicLSTMCellZeroState/Const)target_net/BasicLSTMCellZeroState/Const_1-target_net/BasicLSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
r
-target_net/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'target_net/BasicLSTMCellZeroState/zerosFill(target_net/BasicLSTMCellZeroState/concat-target_net/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:

s
)target_net/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_5Const*
dtype0*
_output_shapes
:*
valueB:

q
/target_net/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
*target_net/BasicLSTMCellZeroState/concat_1ConcatV2)target_net/BasicLSTMCellZeroState/Const_4)target_net/BasicLSTMCellZeroState/Const_5/target_net/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
t
/target_net/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)target_net/BasicLSTMCellZeroState/zeros_1Fill*target_net/BasicLSTMCellZeroState/concat_1/target_net/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

s
)target_net/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
Q
target_net/RankConst*
dtype0*
_output_shapes
: *
value	B :
X
target_net/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
target_net/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
target_net/rangeRangetarget_net/range/starttarget_net/Ranktarget_net/range/delta*
_output_shapes
:*

Tidx0
k
target_net/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
X
target_net/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/concatConcatV2target_net/concat/values_0target_net/rangetarget_net/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
target_net/transpose	Transposetarget_net/Reshape_2target_net/concat*
T0*+
_output_shapes
:���������
*
Tperm0
h
target_net/rnn/ShapeShapetarget_net/transpose*
T0*
out_type0*
_output_shapes
:
l
"target_net/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
n
$target_net/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
n
$target_net/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
target_net/rnn/strided_sliceStridedSlicetarget_net/rnn/Shape"target_net/rnn/strided_slice/stack$target_net/rnn/strided_slice/stack_1$target_net/rnn/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
j
target_net/rnn/Shape_1Shapetarget_net/transpose*
_output_shapes
:*
T0*
out_type0
n
$target_net/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
target_net/rnn/strided_slice_1StridedSlicetarget_net/rnn/Shape_1$target_net/rnn/strided_slice_1/stack&target_net/rnn/strided_slice_1/stack_1&target_net/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
j
target_net/rnn/Shape_2Shapetarget_net/transpose*
T0*
out_type0*
_output_shapes
:
n
$target_net/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
target_net/rnn/strided_slice_2StridedSlicetarget_net/rnn/Shape_2$target_net/rnn/strided_slice_2/stack&target_net/rnn/strided_slice_2/stack_1&target_net/rnn/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
_
target_net/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/ExpandDims
ExpandDimstarget_net/rnn/strided_slice_2target_net/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
^
target_net/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB:

\
target_net/rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
target_net/rnn/concatConcatV2target_net/rnn/ExpandDimstarget_net/rnn/Consttarget_net/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
_
target_net/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
target_net/rnn/zerosFilltarget_net/rnn/concattarget_net/rnn/zeros/Const*
T0*'
_output_shapes
:���������

U
target_net/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/TensorArrayTensorArrayV3target_net/rnn/strided_slice_1*
element_shape:*
dynamic_size( *
clear_after_read(*:
tensor_array_name%#target_net/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
target_net/rnn/TensorArray_1TensorArrayV3target_net/rnn/strided_slice_1*9
tensor_array_name$"target_net/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
{
'target_net/rnn/TensorArrayUnstack/ShapeShapetarget_net/transpose*
_output_shapes
:*
T0*
out_type0

5target_net/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7target_net/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7target_net/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
/target_net/rnn/TensorArrayUnstack/strided_sliceStridedSlice'target_net/rnn/TensorArrayUnstack/Shape5target_net/rnn/TensorArrayUnstack/strided_slice/stack7target_net/rnn/TensorArrayUnstack/strided_slice/stack_17target_net/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
o
-target_net/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
o
-target_net/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
'target_net/rnn/TensorArrayUnstack/rangeRange-target_net/rnn/TensorArrayUnstack/range/start/target_net/rnn/TensorArrayUnstack/strided_slice-target_net/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Itarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3target_net/rnn/TensorArray_1'target_net/rnn/TensorArrayUnstack/rangetarget_net/transposetarget_net/rnn/TensorArray_1:1*
T0*'
_class
loc:@target_net/transpose*
_output_shapes
: 
�
target_net/rnn/while/EnterEntertarget_net/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context
�
target_net/rnn/while/Enter_1Entertarget_net/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context
�
target_net/rnn/while/Enter_2Enter'target_net/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*2

frame_name$"target_net/rnn/while/while_context
�
target_net/rnn/while/Enter_3Enter)target_net/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*2

frame_name$"target_net/rnn/while/while_context
�
target_net/rnn/while/MergeMergetarget_net/rnn/while/Enter"target_net/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
target_net/rnn/while/Merge_1Mergetarget_net/rnn/while/Enter_1$target_net/rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0
�
target_net/rnn/while/Merge_2Mergetarget_net/rnn/while/Enter_2$target_net/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
target_net/rnn/while/Merge_3Mergetarget_net/rnn/while/Enter_3$target_net/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
target_net/rnn/while/Less/EnterEntertarget_net/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context

target_net/rnn/while/LessLesstarget_net/rnn/while/Mergetarget_net/rnn/while/Less/Enter*
_output_shapes
: *
T0
\
target_net/rnn/while/LoopCondLoopCondtarget_net/rnn/while/Less*
_output_shapes
: 
�
target_net/rnn/while/SwitchSwitchtarget_net/rnn/while/Mergetarget_net/rnn/while/LoopCond*
_output_shapes
: : *
T0*-
_class#
!loc:@target_net/rnn/while/Merge
�
target_net/rnn/while/Switch_1Switchtarget_net/rnn/while/Merge_1target_net/rnn/while/LoopCond*
_output_shapes
: : *
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_1
�
target_net/rnn/while/Switch_2Switchtarget_net/rnn/while/Merge_2target_net/rnn/while/LoopCond*(
_output_shapes
:
:
*
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_2
�
target_net/rnn/while/Switch_3Switchtarget_net/rnn/while/Merge_3target_net/rnn/while/LoopCond*
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_3*(
_output_shapes
:
:

i
target_net/rnn/while/IdentityIdentitytarget_net/rnn/while/Switch:1*
T0*
_output_shapes
: 
m
target_net/rnn/while/Identity_1Identitytarget_net/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
u
target_net/rnn/while/Identity_2Identitytarget_net/rnn/while/Switch_2:1*
T0*
_output_shapes

:

u
target_net/rnn/while/Identity_3Identitytarget_net/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
,target_net/rnn/while/TensorArrayReadV3/EnterEntertarget_net/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*2

frame_name$"target_net/rnn/while/while_context
�
.target_net/rnn/while/TensorArrayReadV3/Enter_1EnterItarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context
�
&target_net/rnn/while/TensorArrayReadV3TensorArrayReadV3,target_net/rnn/while/TensorArrayReadV3/Entertarget_net/rnn/while/Identity.target_net/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
Ftarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB"   (   
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB
 *�衾*
dtype0*
_output_shapes
: 
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Ntarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformFtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:(*

seed *
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
seed2 
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSubDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulNtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
@target_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAddDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
%target_net/rnn/basic_lstm_cell/kernel
VariableV2*
shared_name *8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
,target_net/rnn/basic_lstm_cell/kernel/AssignAssign%target_net/rnn/basic_lstm_cell/kernel@target_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
*target_net/rnn/basic_lstm_cell/kernel/readIdentity%target_net/rnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
5target_net/rnn/basic_lstm_cell/bias/Initializer/ConstConst*6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
#target_net/rnn/basic_lstm_cell/bias
VariableV2*
shared_name *6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
*target_net/rnn/basic_lstm_cell/bias/AssignAssign#target_net/rnn/basic_lstm_cell/bias5target_net/rnn/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias
~
(target_net/rnn/basic_lstm_cell/bias/readIdentity#target_net/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
�
4target_net/rnn/while/rnn/basic_lstm_cell/concat/axisConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
/target_net/rnn/while/rnn/basic_lstm_cell/concatConcatV2&target_net/rnn/while/TensorArrayReadV3target_net/rnn/while/Identity_34target_net/rnn/while/rnn/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
5target_net/rnn/while/rnn/basic_lstm_cell/MatMul/EnterEnter*target_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*2

frame_name$"target_net/rnn/while/while_context
�
/target_net/rnn/while/rnn/basic_lstm_cell/MatMulMatMul/target_net/rnn/while/rnn/basic_lstm_cell/concat5target_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter*
_output_shapes

:(*
transpose_a( *
transpose_b( *
T0
�
6target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/EnterEnter(target_net/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*2

frame_name$"target_net/rnn/while/while_context
�
0target_net/rnn/while/rnn/basic_lstm_cell/BiasAddBiasAdd/target_net/rnn/while/rnn/basic_lstm_cell/MatMul6target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes

:(*
T0
�
.target_net/rnn/while/rnn/basic_lstm_cell/ConstConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
8target_net/rnn/while/rnn/basic_lstm_cell/split/split_dimConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
.target_net/rnn/while/rnn/basic_lstm_cell/splitSplit8target_net/rnn/while/rnn/basic_lstm_cell/split/split_dim0target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
.target_net/rnn/while/rnn/basic_lstm_cell/add/yConst^target_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,target_net/rnn/while/rnn/basic_lstm_cell/addAdd0target_net/rnn/while/rnn/basic_lstm_cell/split:2.target_net/rnn/while/rnn/basic_lstm_cell/add/y*
T0*
_output_shapes

:

�
0target_net/rnn/while/rnn/basic_lstm_cell/SigmoidSigmoid,target_net/rnn/while/rnn/basic_lstm_cell/add*
T0*
_output_shapes

:

�
,target_net/rnn/while/rnn/basic_lstm_cell/mulMultarget_net/rnn/while/Identity_20target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1Sigmoid.target_net/rnn/while/rnn/basic_lstm_cell/split*
T0*
_output_shapes

:

�
-target_net/rnn/while/rnn/basic_lstm_cell/TanhTanh0target_net/rnn/while/rnn/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/mul_1Mul2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1-target_net/rnn/while/rnn/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/add_1Add,target_net/rnn/while/rnn/basic_lstm_cell/mul.target_net/rnn/while/rnn/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
/target_net/rnn/while/rnn/basic_lstm_cell/Tanh_1Tanh.target_net/rnn/while/rnn/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2Sigmoid0target_net/rnn/while/rnn/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/mul_2Mul/target_net/rnn/while/rnn/basic_lstm_cell/Tanh_12target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*
_output_shapes

:
*
T0
�
>target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntertarget_net/rnn/TensorArray*
T0*A
_class7
53loc:@target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*2

frame_name$"target_net/rnn/while/while_context
�
8target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3>target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entertarget_net/rnn/while/Identity.target_net/rnn/while/rnn/basic_lstm_cell/mul_2target_net/rnn/while/Identity_1*
T0*A
_class7
53loc:@target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
_output_shapes
: 
|
target_net/rnn/while/add/yConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
{
target_net/rnn/while/addAddtarget_net/rnn/while/Identitytarget_net/rnn/while/add/y*
T0*
_output_shapes
: 
n
"target_net/rnn/while/NextIterationNextIterationtarget_net/rnn/while/add*
_output_shapes
: *
T0
�
$target_net/rnn/while/NextIteration_1NextIteration8target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
$target_net/rnn/while/NextIteration_2NextIteration.target_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
$target_net/rnn/while/NextIteration_3NextIteration.target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
T0*
_output_shapes

:

_
target_net/rnn/while/ExitExittarget_net/rnn/while/Switch*
T0*
_output_shapes
: 
c
target_net/rnn/while/Exit_1Exittarget_net/rnn/while/Switch_1*
_output_shapes
: *
T0
k
target_net/rnn/while/Exit_2Exittarget_net/rnn/while/Switch_2*
T0*
_output_shapes

:

k
target_net/rnn/while/Exit_3Exittarget_net/rnn/while/Switch_3*
_output_shapes

:
*
T0
�
1target_net/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3target_net/rnn/TensorArraytarget_net/rnn/while/Exit_1*-
_class#
!loc:@target_net/rnn/TensorArray*
_output_shapes
: 
�
+target_net/rnn/TensorArrayStack/range/startConst*
value	B : *-
_class#
!loc:@target_net/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
+target_net/rnn/TensorArrayStack/range/deltaConst*
value	B :*-
_class#
!loc:@target_net/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
%target_net/rnn/TensorArrayStack/rangeRange+target_net/rnn/TensorArrayStack/range/start1target_net/rnn/TensorArrayStack/TensorArraySizeV3+target_net/rnn/TensorArrayStack/range/delta*

Tidx0*-
_class#
!loc:@target_net/rnn/TensorArray*#
_output_shapes
:���������
�
3target_net/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3target_net/rnn/TensorArray%target_net/rnn/TensorArrayStack/rangetarget_net/rnn/while/Exit_1*
element_shape
:
*-
_class#
!loc:@target_net/rnn/TensorArray*
dtype0*"
_output_shapes
:

`
target_net/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
U
target_net/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
\
target_net/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
\
target_net/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
target_net/rnn/rangeRangetarget_net/rnn/range/starttarget_net/rnn/Ranktarget_net/rnn/range/delta*
_output_shapes
:*

Tidx0
q
 target_net/rnn/concat_1/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
^
target_net/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/concat_1ConcatV2 target_net/rnn/concat_1/values_0target_net/rnn/rangetarget_net/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
target_net/rnn/transpose	Transpose3target_net/rnn/TensorArrayStack/TensorArrayGatherV3target_net/rnn/concat_1*
T0*"
_output_shapes
:
*
Tperm0
�
target_net/MatMul_1MatMultarget_net/rnn/while/Exit_3target_net/Variable_1/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
q
target_net/add_1Addtarget_net/MatMul_1target_net/Variable_3/read*
_output_shapes

:*
T0
�
AssignAssigntarget_net/Variableeval_net/Variable/read*
use_locking(*
T0*&
_class
loc:@target_net/Variable*
validate_shape(*
_output_shapes

:

�
Assign_1Assigntarget_net/Variable_1eval_net/Variable_1/read*
T0*(
_class
loc:@target_net/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
Assign_2Assigntarget_net/Variable_2eval_net/Variable_2/read*
use_locking(*
T0*(
_class
loc:@target_net/Variable_2*
validate_shape(*
_output_shapes
:

�
Assign_3Assigntarget_net/Variable_3eval_net/Variable_3/read*
T0*(
_class
loc:@target_net/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking("��r�     ��\	�|�9���AJݱ

�+�+
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
ApplyRMSProp
var"T�

ms"T�
mom"T�
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
7
Less
x"T
y"T
z
"
Ttype:
2		
!
LoopCond	
input


output

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
/
Sigmoid
x"T
y"T"
Ttype:	
2
<
SigmoidGrad
y"T
dy"T
z"T"
Ttype:	
2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	�
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype�
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( �
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring �
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
9
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
,
Tanh
x"T
y"T"
Ttype:	
2
9
TanhGrad
y"T
dy"T
z"T"
Ttype:	
2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.4.12
b'unknown'��
d
sPlaceholder*
dtype0*'
_output_shapes
:���������)*
shape:���������)
k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
m
eval_net/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
`
eval_net/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
eval_net/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+eval_net/random_normal/RandomStandardNormalRandomStandardNormaleval_net/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:
*
seed2 
�
eval_net/random_normal/mulMul+eval_net/random_normal/RandomStandardNormaleval_net/random_normal/stddev*
T0*
_output_shapes

:


eval_net/random_normalAddeval_net/random_normal/muleval_net/random_normal/mean*
T0*
_output_shapes

:

�
eval_net/Variable
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
eval_net/Variable/AssignAssigneval_net/Variableeval_net/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*$
_class
loc:@eval_net/Variable
�
eval_net/Variable/readIdentityeval_net/Variable*
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

o
eval_net/random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"
      
b
eval_net/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
eval_net/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
-eval_net/random_normal_1/RandomStandardNormalRandomStandardNormaleval_net/random_normal_1/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
eval_net/random_normal_1/mulMul-eval_net/random_normal_1/RandomStandardNormaleval_net/random_normal_1/stddev*
T0*
_output_shapes

:

�
eval_net/random_normal_1Addeval_net/random_normal_1/muleval_net/random_normal_1/mean*
T0*
_output_shapes

:

�
eval_net/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
eval_net/Variable_1/AssignAssigneval_net/Variable_1eval_net/random_normal_1*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_1*
validate_shape(*
_output_shapes

:

�
eval_net/Variable_1/readIdentityeval_net/Variable_1*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

[
eval_net/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:


eval_net/Variable_2
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
eval_net/Variable_2/AssignAssigneval_net/Variable_2eval_net/Const*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2*
validate_shape(*
_output_shapes
:

�
eval_net/Variable_2/readIdentityeval_net/Variable_2*
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

]
eval_net/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:

eval_net/Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
eval_net/Variable_3/AssignAssigneval_net/Variable_3eval_net/Const_1*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_3*
validate_shape(*
_output_shapes
:
�
eval_net/Variable_3/readIdentityeval_net/Variable_3*
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:
k
eval_net/Reshape/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      
z
eval_net/ReshapeReshapeseval_net/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:���������
i
eval_net/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
eval_net/Reshape_1Reshapeeval_net/Reshapeeval_net/Reshape_1/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
eval_net/MatMulMatMuleval_net/Reshape_1eval_net/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
p
eval_net/addAddeval_net/MatMuleval_net/Variable_2/read*
T0*'
_output_shapes
:���������

m
eval_net/Reshape_2/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
eval_net/Reshape_2Reshapeeval_net/addeval_net/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������

o
%eval_net/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
m
+eval_net/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
&eval_net/BasicLSTMCellZeroState/concatConcatV2%eval_net/BasicLSTMCellZeroState/Const'eval_net/BasicLSTMCellZeroState/Const_1+eval_net/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
p
+eval_net/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%eval_net/BasicLSTMCellZeroState/zerosFill&eval_net/BasicLSTMCellZeroState/concat+eval_net/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:
*
T0
q
'eval_net/BasicLSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
q
'eval_net/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

q
'eval_net/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
o
-eval_net/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
(eval_net/BasicLSTMCellZeroState/concat_1ConcatV2'eval_net/BasicLSTMCellZeroState/Const_4'eval_net/BasicLSTMCellZeroState/Const_5-eval_net/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
r
-eval_net/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'eval_net/BasicLSTMCellZeroState/zeros_1Fill(eval_net/BasicLSTMCellZeroState/concat_1-eval_net/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

q
'eval_net/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

O
eval_net/RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
eval_net/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
V
eval_net/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
z
eval_net/rangeRangeeval_net/range/starteval_net/Rankeval_net/range/delta*
_output_shapes
:*

Tidx0
i
eval_net/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
V
eval_net/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/concatConcatV2eval_net/concat/values_0eval_net/rangeeval_net/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
eval_net/transpose	Transposeeval_net/Reshape_2eval_net/concat*
T0*+
_output_shapes
:���������
*
Tperm0
d
eval_net/rnn/ShapeShapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
j
 eval_net/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
l
"eval_net/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"eval_net/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/rnn/strided_sliceStridedSliceeval_net/rnn/Shape eval_net/rnn/strided_slice/stack"eval_net/rnn/strided_slice/stack_1"eval_net/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
eval_net/rnn/Shape_1Shapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
l
"eval_net/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$eval_net/rnn/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
n
$eval_net/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
eval_net/rnn/strided_slice_1StridedSliceeval_net/rnn/Shape_1"eval_net/rnn/strided_slice_1/stack$eval_net/rnn/strided_slice_1/stack_1$eval_net/rnn/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
f
eval_net/rnn/Shape_2Shapeeval_net/transpose*
T0*
out_type0*
_output_shapes
:
l
"eval_net/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
n
$eval_net/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
n
$eval_net/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/rnn/strided_slice_2StridedSliceeval_net/rnn/Shape_2"eval_net/rnn/strided_slice_2/stack$eval_net/rnn/strided_slice_2/stack_1$eval_net/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
]
eval_net/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/ExpandDims
ExpandDimseval_net/rnn/strided_slice_2eval_net/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
\
eval_net/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Z
eval_net/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/concatConcatV2eval_net/rnn/ExpandDimseval_net/rnn/Consteval_net/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
]
eval_net/rnn/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
{
eval_net/rnn/zerosFilleval_net/rnn/concateval_net/rnn/zeros/Const*
T0*'
_output_shapes
:���������

S
eval_net/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/rnn/TensorArrayTensorArrayV3eval_net/rnn/strided_slice_1*8
tensor_array_name#!eval_net/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
�
eval_net/rnn/TensorArray_1TensorArrayV3eval_net/rnn/strided_slice_1*7
tensor_array_name" eval_net/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
w
%eval_net/rnn/TensorArrayUnstack/ShapeShapeeval_net/transpose*
_output_shapes
:*
T0*
out_type0
}
3eval_net/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
-eval_net/rnn/TensorArrayUnstack/strided_sliceStridedSlice%eval_net/rnn/TensorArrayUnstack/Shape3eval_net/rnn/TensorArrayUnstack/strided_slice/stack5eval_net/rnn/TensorArrayUnstack/strided_slice/stack_15eval_net/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
m
+eval_net/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
m
+eval_net/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
%eval_net/rnn/TensorArrayUnstack/rangeRange+eval_net/rnn/TensorArrayUnstack/range/start-eval_net/rnn/TensorArrayUnstack/strided_slice+eval_net/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Geval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3eval_net/rnn/TensorArray_1%eval_net/rnn/TensorArrayUnstack/rangeeval_net/transposeeval_net/rnn/TensorArray_1:1*
T0*%
_class
loc:@eval_net/transpose*
_output_shapes
: 
�
eval_net/rnn/while/EnterEntereval_net/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
eval_net/rnn/while/Enter_1Entereval_net/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
eval_net/rnn/while/Enter_2Enter%eval_net/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant( 
�
eval_net/rnn/while/Enter_3Enter'eval_net/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
_output_shapes

:
*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant( 
�
eval_net/rnn/while/MergeMergeeval_net/rnn/while/Enter eval_net/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
eval_net/rnn/while/Merge_1Mergeeval_net/rnn/while/Enter_1"eval_net/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
eval_net/rnn/while/Merge_2Mergeeval_net/rnn/while/Enter_2"eval_net/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
eval_net/rnn/while/Merge_3Mergeeval_net/rnn/while/Enter_3"eval_net/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
eval_net/rnn/while/Less/EnterEntereval_net/rnn/strided_slice_1*
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
y
eval_net/rnn/while/LessLesseval_net/rnn/while/Mergeeval_net/rnn/while/Less/Enter*
T0*
_output_shapes
: 
X
eval_net/rnn/while/LoopCondLoopCondeval_net/rnn/while/Less*
_output_shapes
: 
�
eval_net/rnn/while/SwitchSwitcheval_net/rnn/while/Mergeeval_net/rnn/while/LoopCond*
T0*+
_class!
loc:@eval_net/rnn/while/Merge*
_output_shapes
: : 
�
eval_net/rnn/while/Switch_1Switcheval_net/rnn/while/Merge_1eval_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_1*
_output_shapes
: : 
�
eval_net/rnn/while/Switch_2Switcheval_net/rnn/while/Merge_2eval_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_2*(
_output_shapes
:
:

�
eval_net/rnn/while/Switch_3Switcheval_net/rnn/while/Merge_3eval_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@eval_net/rnn/while/Merge_3*(
_output_shapes
:
:

e
eval_net/rnn/while/IdentityIdentityeval_net/rnn/while/Switch:1*
_output_shapes
: *
T0
i
eval_net/rnn/while/Identity_1Identityeval_net/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
q
eval_net/rnn/while/Identity_2Identityeval_net/rnn/while/Switch_2:1*
_output_shapes

:
*
T0
q
eval_net/rnn/while/Identity_3Identityeval_net/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
*eval_net/rnn/while/TensorArrayReadV3/EnterEntereval_net/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
,eval_net/rnn/while/TensorArrayReadV3/Enter_1EnterGeval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
�
$eval_net/rnn/while/TensorArrayReadV3TensorArrayReadV3*eval_net/rnn/while/TensorArrayReadV3/Entereval_net/rnn/while/Identity,eval_net/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
Deval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB
 *�衾*
dtype0*
_output_shapes
: 
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Leval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformDeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed *
T0
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSubBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
Beval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulLeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
>eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAddBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulBeval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
#eval_net/rnn/basic_lstm_cell/kernel
VariableV2*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
*eval_net/rnn/basic_lstm_cell/kernel/AssignAssign#eval_net/rnn/basic_lstm_cell/kernel>eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
(eval_net/rnn/basic_lstm_cell/kernel/readIdentity#eval_net/rnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
3eval_net/rnn/basic_lstm_cell/bias/Initializer/ConstConst*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
!eval_net/rnn/basic_lstm_cell/bias
VariableV2*
shape:(*
dtype0*
_output_shapes
:(*
shared_name *4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container 
�
(eval_net/rnn/basic_lstm_cell/bias/AssignAssign!eval_net/rnn/basic_lstm_cell/bias3eval_net/rnn/basic_lstm_cell/bias/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
z
&eval_net/rnn/basic_lstm_cell/bias/readIdentity!eval_net/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
�
2eval_net/rnn/while/rnn/basic_lstm_cell/concat/axisConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
-eval_net/rnn/while/rnn/basic_lstm_cell/concatConcatV2$eval_net/rnn/while/TensorArrayReadV3eval_net/rnn/while/Identity_32eval_net/rnn/while/rnn/basic_lstm_cell/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
�
3eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/EnterEnter(eval_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*0

frame_name" eval_net/rnn/while/while_context
�
-eval_net/rnn/while/rnn/basic_lstm_cell/MatMulMatMul-eval_net/rnn/while/rnn/basic_lstm_cell/concat3eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
4eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/EnterEnter&eval_net/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*0

frame_name" eval_net/rnn/while/while_context
�
.eval_net/rnn/while/rnn/basic_lstm_cell/BiasAddBiasAdd-eval_net/rnn/while/rnn/basic_lstm_cell/MatMul4eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
,eval_net/rnn/while/rnn/basic_lstm_cell/ConstConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
6eval_net/rnn/while/rnn/basic_lstm_cell/split/split_dimConst^eval_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
,eval_net/rnn/while/rnn/basic_lstm_cell/splitSplit6eval_net/rnn/while/rnn/basic_lstm_cell/split/split_dim.eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd*<
_output_shapes*
(:
:
:
:
*
	num_split*
T0
�
,eval_net/rnn/while/rnn/basic_lstm_cell/add/yConst^eval_net/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*eval_net/rnn/while/rnn/basic_lstm_cell/addAdd.eval_net/rnn/while/rnn/basic_lstm_cell/split:2,eval_net/rnn/while/rnn/basic_lstm_cell/add/y*
_output_shapes

:
*
T0
�
.eval_net/rnn/while/rnn/basic_lstm_cell/SigmoidSigmoid*eval_net/rnn/while/rnn/basic_lstm_cell/add*
T0*
_output_shapes

:

�
*eval_net/rnn/while/rnn/basic_lstm_cell/mulMuleval_net/rnn/while/Identity_2.eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1Sigmoid,eval_net/rnn/while/rnn/basic_lstm_cell/split*
T0*
_output_shapes

:

�
+eval_net/rnn/while/rnn/basic_lstm_cell/TanhTanh.eval_net/rnn/while/rnn/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
,eval_net/rnn/while/rnn/basic_lstm_cell/mul_1Mul0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1+eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
,eval_net/rnn/while/rnn/basic_lstm_cell/add_1Add*eval_net/rnn/while/rnn/basic_lstm_cell/mul,eval_net/rnn/while/rnn/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1Tanh,eval_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2Sigmoid.eval_net/rnn/while/rnn/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2Mul-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_10eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
<eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntereval_net/rnn/TensorArray*
T0*?
_class5
31loc:@eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
6eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3<eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/rnn/while/Identity,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2eval_net/rnn/while/Identity_1*
T0*?
_class5
31loc:@eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
_output_shapes
: 
x
eval_net/rnn/while/add/yConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
eval_net/rnn/while/addAddeval_net/rnn/while/Identityeval_net/rnn/while/add/y*
T0*
_output_shapes
: 
j
 eval_net/rnn/while/NextIterationNextIterationeval_net/rnn/while/add*
_output_shapes
: *
T0
�
"eval_net/rnn/while/NextIteration_1NextIteration6eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
�
"eval_net/rnn/while/NextIteration_2NextIteration,eval_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
"eval_net/rnn/while/NextIteration_3NextIteration,eval_net/rnn/while/rnn/basic_lstm_cell/mul_2*
T0*
_output_shapes

:

[
eval_net/rnn/while/ExitExiteval_net/rnn/while/Switch*
_output_shapes
: *
T0
_
eval_net/rnn/while/Exit_1Exiteval_net/rnn/while/Switch_1*
T0*
_output_shapes
: 
g
eval_net/rnn/while/Exit_2Exiteval_net/rnn/while/Switch_2*
T0*
_output_shapes

:

g
eval_net/rnn/while/Exit_3Exiteval_net/rnn/while/Switch_3*
T0*
_output_shapes

:

�
/eval_net/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3eval_net/rnn/TensorArrayeval_net/rnn/while/Exit_1*+
_class!
loc:@eval_net/rnn/TensorArray*
_output_shapes
: 
�
)eval_net/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *+
_class!
loc:@eval_net/rnn/TensorArray
�
)eval_net/rnn/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@eval_net/rnn/TensorArray
�
#eval_net/rnn/TensorArrayStack/rangeRange)eval_net/rnn/TensorArrayStack/range/start/eval_net/rnn/TensorArrayStack/TensorArraySizeV3)eval_net/rnn/TensorArrayStack/range/delta*+
_class!
loc:@eval_net/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
1eval_net/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3eval_net/rnn/TensorArray#eval_net/rnn/TensorArrayStack/rangeeval_net/rnn/while/Exit_1*+
_class!
loc:@eval_net/rnn/TensorArray*
dtype0*"
_output_shapes
:
*
element_shape
:

^
eval_net/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
S
eval_net/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Z
eval_net/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Z
eval_net/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_net/rnn/rangeRangeeval_net/rnn/range/starteval_net/rnn/Rankeval_net/rnn/range/delta*
_output_shapes
:*

Tidx0
o
eval_net/rnn/concat_1/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
\
eval_net/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/rnn/concat_1ConcatV2eval_net/rnn/concat_1/values_0eval_net/rnn/rangeeval_net/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
eval_net/rnn/transpose	Transpose1eval_net/rnn/TensorArrayStack/TensorArrayGatherV3eval_net/rnn/concat_1*
Tperm0*
T0*"
_output_shapes
:

�
eval_net/MatMul_1MatMuleval_net/rnn/while/Exit_3eval_net/Variable_1/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
k
eval_net/add_1Addeval_net/MatMul_1eval_net/Variable_3/read*
T0*
_output_shapes

:
w
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/add_1*
T0*'
_output_shapes
:���������
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	loss/MeanMeanloss/SquaredDifference
loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
train/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
Y
train/gradients/f_countConst*
dtype0*
_output_shapes
: *
value	B : 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" eval_net/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
train/gradients/SwitchSwitchtrain/gradients/Mergeeval_net/rnn/while/LoopCond*
T0*
_output_shapes
: : 
u
train/gradients/Add/yConst^eval_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/AddR^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPushV2T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2P^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPushV2R^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPushV2R^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPushV2T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2X^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2V^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2X^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1j^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2*
_output_shapes
: *
T0
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
T0*
_output_shapes
: 
Y
train/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
train/gradients/Merge_1Mergetrain/gradients/b_count_1train/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
"train/gradients/GreaterEqual/EnterEntertrain/gradients/b_count*
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
train/gradients/GreaterEqualGreaterEqualtrain/gradients/Merge_1"train/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
[
train/gradients/b_count_2LoopCondtrain/gradients/GreaterEqual*
_output_shapes
: 
y
train/gradients/Switch_1Switchtrain/gradients/Merge_1train/gradients/b_count_2*
T0*
_output_shapes
: : 
{
train/gradients/SubSubtrain/gradients/Switch_1:1"train/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
train/gradients/NextIteration_1NextIterationtrain/gradients/SubM^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/b_sync*
T0*
_output_shapes
: 
\
train/gradients/b_count_3Exittrain/gradients/Switch_1*
_output_shapes
: *
T0
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/loss/Mean_grad/ShapeShapeloss/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*'
_output_shapes
:���������*
T0
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/loss/SquaredDifference_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/add_1'^train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:
�
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
�
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg*
_output_shapes

:
z
)train/gradients/eval_net/add_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
u
+train/gradients/eval_net/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
9train/gradients/eval_net/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs)train/gradients/eval_net/add_1_grad/Shape+train/gradients/eval_net/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
'train/gradients/eval_net/add_1_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_19train/gradients/eval_net/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
+train/gradients/eval_net/add_1_grad/ReshapeReshape'train/gradients/eval_net/add_1_grad/Sum)train/gradients/eval_net/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:
�
)train/gradients/eval_net/add_1_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1;train/gradients/eval_net/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
-train/gradients/eval_net/add_1_grad/Reshape_1Reshape)train/gradients/eval_net/add_1_grad/Sum_1+train/gradients/eval_net/add_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
4train/gradients/eval_net/add_1_grad/tuple/group_depsNoOp,^train/gradients/eval_net/add_1_grad/Reshape.^train/gradients/eval_net/add_1_grad/Reshape_1
�
<train/gradients/eval_net/add_1_grad/tuple/control_dependencyIdentity+train/gradients/eval_net/add_1_grad/Reshape5^train/gradients/eval_net/add_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/add_1_grad/Reshape*
_output_shapes

:
�
>train/gradients/eval_net/add_1_grad/tuple/control_dependency_1Identity-train/gradients/eval_net/add_1_grad/Reshape_15^train/gradients/eval_net/add_1_grad/tuple/group_deps*
_output_shapes
:*
T0*@
_class6
42loc:@train/gradients/eval_net/add_1_grad/Reshape_1
�
-train/gradients/eval_net/MatMul_1_grad/MatMulMatMul<train/gradients/eval_net/add_1_grad/tuple/control_dependencyeval_net/Variable_1/read*
transpose_b(*
T0*
_output_shapes

:
*
transpose_a( 
�
/train/gradients/eval_net/MatMul_1_grad/MatMul_1MatMuleval_net/rnn/while/Exit_3<train/gradients/eval_net/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
7train/gradients/eval_net/MatMul_1_grad/tuple/group_depsNoOp.^train/gradients/eval_net/MatMul_1_grad/MatMul0^train/gradients/eval_net/MatMul_1_grad/MatMul_1
�
?train/gradients/eval_net/MatMul_1_grad/tuple/control_dependencyIdentity-train/gradients/eval_net/MatMul_1_grad/MatMul8^train/gradients/eval_net/MatMul_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/eval_net/MatMul_1_grad/MatMul*
_output_shapes

:

�
Atrain/gradients/eval_net/MatMul_1_grad/tuple/control_dependency_1Identity/train/gradients/eval_net/MatMul_1_grad/MatMul_18^train/gradients/eval_net/MatMul_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*B
_class8
64loc:@train/gradients/eval_net/MatMul_1_grad/MatMul_1
Z
train/gradients/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
train/gradients/zeros_1Const*
dtype0*
_output_shapes

:
*
valueB
*    
�
5train/gradients/eval_net/rnn/while/Exit_3_grad/b_exitEnter?train/gradients/eval_net/MatMul_1_grad/tuple/control_dependency*
parallel_iterations *
_output_shapes

:
*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant( 
�
5train/gradients/eval_net/rnn/while/Exit_1_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
5train/gradients/eval_net/rnn/while/Exit_2_grad/b_exitEntertrain/gradients/zeros_1*
parallel_iterations *
_output_shapes

:
*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant( 
�
9train/gradients/eval_net/rnn/while/Switch_3_grad/b_switchMerge5train/gradients/eval_net/rnn/while/Exit_3_grad/b_exit@train/gradients/eval_net/rnn/while/Switch_3_grad_1/NextIteration*
N* 
_output_shapes
:
: *
T0
�
9train/gradients/eval_net/rnn/while/Switch_2_grad/b_switchMerge5train/gradients/eval_net/rnn/while/Exit_2_grad/b_exit@train/gradients/eval_net/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
6train/gradients/eval_net/rnn/while/Merge_3_grad/SwitchSwitch9train/gradients/eval_net/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

�
@train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_depsNoOp7^train/gradients/eval_net/rnn/while/Merge_3_grad/Switch
�
Htrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity6train/gradients/eval_net/rnn/while/Merge_3_grad/SwitchA^train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Jtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity8train/gradients/eval_net/rnn/while/Merge_3_grad/Switch:1A^train/gradients/eval_net/rnn/while/Merge_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_3_grad/b_switch
�
6train/gradients/eval_net/rnn/while/Merge_2_grad/SwitchSwitch9train/gradients/eval_net/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*(
_output_shapes
:
:

�
@train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_depsNoOp7^train/gradients/eval_net/rnn/while/Merge_2_grad/Switch
�
Htrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity6train/gradients/eval_net/rnn/while/Merge_2_grad/SwitchA^train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
Jtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity8train/gradients/eval_net/rnn/while/Merge_2_grad/Switch:1A^train/gradients/eval_net/rnn/while/Merge_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
4train/gradients/eval_net/rnn/while/Enter_3_grad/ExitExitHtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
4train/gradients/eval_net/rnn/while/Enter_2_grad/ExitExitHtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
valueB :
���������*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/Enter0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/b_syncControlTriggerQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2O^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2W^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2U^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2W^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1i^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mulMulJtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/SumSumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mulWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1*
dtype0*
_output_shapes
: 
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
	elem_type0*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/Enter-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1MulRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Jtrain/gradients/eval_net/rnn/while/Merge_3_grad/tuple/control_dependency_1*
_output_shapes

:
*
T0
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Sum_1SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:

�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:

�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1*
_output_shapes

:
*
T0
�
train/gradients/AddNAddNJtrain/gradients/eval_net/rnn/while/Merge_2_grad/tuple/control_dependency_1Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*L
_classB
@>loc:@train/gradients/eval_net/rnn/while/Switch_2_grad/b_switch*
N*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/SumSumtrain/gradients/AddNWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Sum_1Sumtrain/gradients/AddNYtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:

�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ShapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*A
_class7
53loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
dtype0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_accStackV2Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc/max_size*A
_class7
53loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/EnterEnterItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/Enter.eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mulMulZtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependencyNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/SumSumCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mulUtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeReshapeCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*0
_class&
$"loc:@eval_net/rnn/while/Identity_2*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*0
_class&
$"loc:@eval_net/rnn/while/Identity_2*

stack_name *
_output_shapes
:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/Entereval_net/rnn/while/Identity_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1MulPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Sum_1SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1ReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Sum_1Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_depsNoOpH^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependencyIdentityGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/ReshapeQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:

�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ShapeItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
valueB :
���������*>
_class4
20loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*
dtype0*
_output_shapes
: 
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_accStackV2Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*>
_class4
20loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/Enter+eval_net/rnn/while/rnn/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mulMul\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/SumSumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mulWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
	elem_type0*C
_class9
75loc:@eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context*
T0*
is_constant(
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/Enter0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnterMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1MulRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Sum_1SumGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Sum_1Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
Rtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/ReshapeS^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:

�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1S^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:

�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_grad/TanhGradTanhGradPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
@train/gradients/eval_net/rnn/while/Switch_2_grad_1/NextIterationNextIterationXtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ShapeGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/SumSumOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradUtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeReshapeCtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/SumEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
Etrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Sum_1SumOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradWtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1ReshapeEtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Sum_1Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_depsNoOpH^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependencyIdentityGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/ReshapeQ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape*
_output_shapes

:

�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1Q^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatConcatV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_grad/TanhGradXtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/add_grad/tuple/control_dependencyQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*
_output_shapes

:(
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes
:(*
T0
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpI^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatP^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concatU^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradU^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter(eval_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulMatMul\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul/Enter*
_output_shapes

:*
transpose_a( *
transpose_b(*
T0
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
valueB :
���������*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/concat*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*@
_class6
42loc:@eval_net/rnn/while/rnn/basic_lstm_cell/concat
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/Enter-eval_net/rnn/while/rnn/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:*
	elem_type0
�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1MatMulVtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulL^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
�
[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMulT^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:(*
T0*^
_classT
RPloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
_output_shapes
:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant( 
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes

:(: *
T0
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/AddAddRtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationMtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
Gtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/mod/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Ftrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/modFloorModLtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/mod/ConstGtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeShape$eval_net/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_sizeConst*
valueB :
���������*7
_class-
+)loc:@eval_net/rnn/while/TensorArrayReadV3*
dtype0*
_output_shapes
: 
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Xtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_size*7
_class-
+)loc:@eval_net/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/EnterEnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter$eval_net/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ttrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*'
_output_shapes
:���������
*
	elem_type0
�
Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_sizeConst*
valueB :
���������*0
_class&
$"loc:@eval_net/rnn/while/Identity_3*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Ztrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_size*0
_class&
$"loc:@eval_net/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1eval_net/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2\train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Itrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeNShapeNTtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetFtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/modItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeNKtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
Htrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceSlice[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffsetItrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:������������������
�
Jtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1Slice[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ConcatOffset:1Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:������������������*
Index0*
T0
�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_depsNoOpI^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceK^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1
�
[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityHtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/SliceT^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityJtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1T^train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*]
_classS
QOloc:@train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
Ntrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0*
_output_shapes

:(*
valueB(*    
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterNtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergePtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Ltrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/AddAddQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Switch:1]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:(*
T0
�
Vtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationLtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Add*
_output_shapes

:(*
T0
�
Ptrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitOtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
atrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntereval_net/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterGeval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context
�
[train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3atrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^train/gradients/Sub*
sourcetrain/gradients*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
�
Wtrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1\^train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*=
_class3
1/loc:@eval_net/rnn/while/TensorArrayReadV3/Enter
�
ltrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc/max_sizeConst*
valueB :
���������*.
_class$
" loc:@eval_net/rnn/while/Identity*
dtype0*
_output_shapes
: 
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStackV2ltrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc/max_size*.
_class$
" loc:@eval_net/rnn/while/Identity*

stack_name *
_output_shapes
:*
	elem_type0
�
ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/EnterEnterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" eval_net/rnn/while/while_context
�
itrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2StackPushV2ctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/rnn/while/Identity^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
ntrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2/EnterEnterctrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc*
parallel_iterations *
_output_shapes
:*@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant(
�
htrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2
StackPopV2ntrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
]train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3[train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3htrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2[train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependencyWtrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
�
Gtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterGtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
_output_shapes
: *@

frame_name20train/gradients/eval_net/rnn/while/while_context*
T0*
is_constant( 
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Otrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Htrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2train/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Etrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddJtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1]train/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Otrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationEtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitHtrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
@train/gradients/eval_net/rnn/while/Switch_3_grad_1/NextIterationNextIteration]train/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
~train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3eval_net/rnn/TensorArray_1Itrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *
sourcetrain/gradients*-
_class#
!loc:@eval_net/rnn/TensorArray_1
�
ztrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*-
_class#
!loc:@eval_net/rnn/TensorArray_1*
_output_shapes
: 
�
ptrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3~train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3%eval_net/rnn/TensorArrayUnstack/rangeztrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*
_output_shapes
:
�
mtrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpJ^train/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3q^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
utrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityptrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3n^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*+
_output_shapes
:���������
*
T0*�
_classy
wuloc:@train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
wtrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityItrain/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3n^train/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*\
_classR
PNloc:@train/gradients/eval_net/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
9train/gradients/eval_net/transpose_grad/InvertPermutationInvertPermutationeval_net/concat*
_output_shapes
:*
T0
�
1train/gradients/eval_net/transpose_grad/transpose	Transposeutrain/gradients/eval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency9train/gradients/eval_net/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:���������

y
-train/gradients/eval_net/Reshape_2_grad/ShapeShapeeval_net/add*
T0*
out_type0*
_output_shapes
:
�
/train/gradients/eval_net/Reshape_2_grad/ReshapeReshape1train/gradients/eval_net/transpose_grad/transpose-train/gradients/eval_net/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

v
'train/gradients/eval_net/add_grad/ShapeShapeeval_net/MatMul*
T0*
out_type0*
_output_shapes
:
s
)train/gradients/eval_net/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
7train/gradients/eval_net/add_grad/BroadcastGradientArgsBroadcastGradientArgs'train/gradients/eval_net/add_grad/Shape)train/gradients/eval_net/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
%train/gradients/eval_net/add_grad/SumSum/train/gradients/eval_net/Reshape_2_grad/Reshape7train/gradients/eval_net/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
)train/gradients/eval_net/add_grad/ReshapeReshape%train/gradients/eval_net/add_grad/Sum'train/gradients/eval_net/add_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
'train/gradients/eval_net/add_grad/Sum_1Sum/train/gradients/eval_net/Reshape_2_grad/Reshape9train/gradients/eval_net/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
+train/gradients/eval_net/add_grad/Reshape_1Reshape'train/gradients/eval_net/add_grad/Sum_1)train/gradients/eval_net/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
2train/gradients/eval_net/add_grad/tuple/group_depsNoOp*^train/gradients/eval_net/add_grad/Reshape,^train/gradients/eval_net/add_grad/Reshape_1
�
:train/gradients/eval_net/add_grad/tuple/control_dependencyIdentity)train/gradients/eval_net/add_grad/Reshape3^train/gradients/eval_net/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/eval_net/add_grad/Reshape*'
_output_shapes
:���������

�
<train/gradients/eval_net/add_grad/tuple/control_dependency_1Identity+train/gradients/eval_net/add_grad/Reshape_13^train/gradients/eval_net/add_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/add_grad/Reshape_1*
_output_shapes
:

�
+train/gradients/eval_net/MatMul_grad/MatMulMatMul:train/gradients/eval_net/add_grad/tuple/control_dependencyeval_net/Variable/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
-train/gradients/eval_net/MatMul_grad/MatMul_1MatMuleval_net/Reshape_1:train/gradients/eval_net/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
5train/gradients/eval_net/MatMul_grad/tuple/group_depsNoOp,^train/gradients/eval_net/MatMul_grad/MatMul.^train/gradients/eval_net/MatMul_grad/MatMul_1
�
=train/gradients/eval_net/MatMul_grad/tuple/control_dependencyIdentity+train/gradients/eval_net/MatMul_grad/MatMul6^train/gradients/eval_net/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/eval_net/MatMul_grad/MatMul*'
_output_shapes
:���������
�
?train/gradients/eval_net/MatMul_grad/tuple/control_dependency_1Identity-train/gradients/eval_net/MatMul_grad/MatMul_16^train/gradients/eval_net/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/eval_net/MatMul_grad/MatMul_1*
_output_shapes

:

�
0train/eval_net/Variable/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*$
_class
loc:@eval_net/Variable*
valueB
*  �?
�
train/eval_net/Variable/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *$
_class
loc:@eval_net/Variable*
	container *
shape
:

�
&train/eval_net/Variable/RMSProp/AssignAssigntrain/eval_net/Variable/RMSProp0train/eval_net/Variable/RMSProp/Initializer/ones*
use_locking(*
T0*$
_class
loc:@eval_net/Variable*
validate_shape(*
_output_shapes

:

�
$train/eval_net/Variable/RMSProp/readIdentitytrain/eval_net/Variable/RMSProp*
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

�
3train/eval_net/Variable/RMSProp_1/Initializer/zerosConst*$
_class
loc:@eval_net/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
!train/eval_net/Variable/RMSProp_1
VariableV2*
shared_name *$
_class
loc:@eval_net/Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
(train/eval_net/Variable/RMSProp_1/AssignAssign!train/eval_net/Variable/RMSProp_13train/eval_net/Variable/RMSProp_1/Initializer/zeros*
T0*$
_class
loc:@eval_net/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
&train/eval_net/Variable/RMSProp_1/readIdentity!train/eval_net/Variable/RMSProp_1*
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

�
2train/eval_net/Variable_1/RMSProp/Initializer/onesConst*&
_class
loc:@eval_net/Variable_1*
valueB
*  �?*
dtype0*
_output_shapes

:

�
!train/eval_net/Variable_1/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *&
_class
loc:@eval_net/Variable_1*
	container *
shape
:

�
(train/eval_net/Variable_1/RMSProp/AssignAssign!train/eval_net/Variable_1/RMSProp2train/eval_net/Variable_1/RMSProp/Initializer/ones*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_1*
validate_shape(*
_output_shapes

:

�
&train/eval_net/Variable_1/RMSProp/readIdentity!train/eval_net/Variable_1/RMSProp*
_output_shapes

:
*
T0*&
_class
loc:@eval_net/Variable_1
�
5train/eval_net/Variable_1/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*&
_class
loc:@eval_net/Variable_1*
valueB
*    
�
#train/eval_net/Variable_1/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *&
_class
loc:@eval_net/Variable_1*
	container *
shape
:

�
*train/eval_net/Variable_1/RMSProp_1/AssignAssign#train/eval_net/Variable_1/RMSProp_15train/eval_net/Variable_1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_1*
validate_shape(*
_output_shapes

:

�
(train/eval_net/Variable_1/RMSProp_1/readIdentity#train/eval_net/Variable_1/RMSProp_1*
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

�
2train/eval_net/Variable_2/RMSProp/Initializer/onesConst*&
_class
loc:@eval_net/Variable_2*
valueB
*  �?*
dtype0*
_output_shapes
:

�
!train/eval_net/Variable_2/RMSProp
VariableV2*
shared_name *&
_class
loc:@eval_net/Variable_2*
	container *
shape:
*
dtype0*
_output_shapes
:

�
(train/eval_net/Variable_2/RMSProp/AssignAssign!train/eval_net/Variable_2/RMSProp2train/eval_net/Variable_2/RMSProp/Initializer/ones*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2*
validate_shape(*
_output_shapes
:

�
&train/eval_net/Variable_2/RMSProp/readIdentity!train/eval_net/Variable_2/RMSProp*
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

�
5train/eval_net/Variable_2/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes
:
*&
_class
loc:@eval_net/Variable_2*
valueB
*    
�
#train/eval_net/Variable_2/RMSProp_1
VariableV2*
shared_name *&
_class
loc:@eval_net/Variable_2*
	container *
shape:
*
dtype0*
_output_shapes
:

�
*train/eval_net/Variable_2/RMSProp_1/AssignAssign#train/eval_net/Variable_2/RMSProp_15train/eval_net/Variable_2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_2*
validate_shape(*
_output_shapes
:

�
(train/eval_net/Variable_2/RMSProp_1/readIdentity#train/eval_net/Variable_2/RMSProp_1*
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

�
2train/eval_net/Variable_3/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes
:*&
_class
loc:@eval_net/Variable_3*
valueB*  �?
�
!train/eval_net/Variable_3/RMSProp
VariableV2*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@eval_net/Variable_3*
	container *
shape:
�
(train/eval_net/Variable_3/RMSProp/AssignAssign!train/eval_net/Variable_3/RMSProp2train/eval_net/Variable_3/RMSProp/Initializer/ones*
T0*&
_class
loc:@eval_net/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
�
&train/eval_net/Variable_3/RMSProp/readIdentity!train/eval_net/Variable_3/RMSProp*
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:
�
5train/eval_net/Variable_3/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*&
_class
loc:@eval_net/Variable_3*
valueB*    
�
#train/eval_net/Variable_3/RMSProp_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@eval_net/Variable_3*
	container *
shape:
�
*train/eval_net/Variable_3/RMSProp_1/AssignAssign#train/eval_net/Variable_3/RMSProp_15train/eval_net/Variable_3/RMSProp_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@eval_net/Variable_3*
validate_shape(*
_output_shapes
:
�
(train/eval_net/Variable_3/RMSProp_1/readIdentity#train/eval_net/Variable_3/RMSProp_1*
_output_shapes
:*
T0*&
_class
loc:@eval_net/Variable_3
�
Btrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Initializer/onesConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB(*  �?*
dtype0*
_output_shapes

:(
�
1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container 
�
8train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/AssignAssign1train/eval_net/rnn/basic_lstm_cell/kernel/RMSPropBtrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Initializer/ones*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
6train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/readIdentity1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
Etrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Initializer/zerosConst*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
	container *
shape
:(
�
:train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/AssignAssign3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1Etrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel
�
8train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/readIdentity3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1*
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
@train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Initializer/onesConst*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*  �?*
dtype0*
_output_shapes
:(
�
/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp
VariableV2*
shared_name *4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
6train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/AssignAssign/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp@train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Initializer/ones*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
4train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/readIdentity/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp*
_output_shapes
:(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias
�
Ctrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
valueB(*    
�
1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1
VariableV2*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name 
�
8train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/AssignAssign1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1Ctrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
6train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/readIdentity1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1*
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
_output_shapes
:(
`
train/RMSProp/learning_rateConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
X
train/RMSProp/decayConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
[
train/RMSProp/momentumConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
3train/RMSProp/update_eval_net/Variable/ApplyRMSPropApplyRMSPropeval_net/Variabletrain/eval_net/Variable/RMSProp!train/eval_net/Variable/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@eval_net/Variable*
_output_shapes

:

�
5train/RMSProp/update_eval_net/Variable_1/ApplyRMSPropApplyRMSPropeval_net/Variable_1!train/eval_net/Variable_1/RMSProp#train/eval_net/Variable_1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonAtrain/gradients/eval_net/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@eval_net/Variable_1*
_output_shapes

:

�
5train/RMSProp/update_eval_net/Variable_2/ApplyRMSPropApplyRMSPropeval_net/Variable_2!train/eval_net/Variable_2/RMSProp#train/eval_net/Variable_2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon<train/gradients/eval_net/add_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@eval_net/Variable_2*
_output_shapes
:

�
5train/RMSProp/update_eval_net/Variable_3/ApplyRMSPropApplyRMSPropeval_net/Variable_3!train/eval_net/Variable_3/RMSProp#train/eval_net/Variable_3/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon>train/gradients/eval_net/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@eval_net/Variable_3*
_output_shapes
:
�
Etrain/RMSProp/update_eval_net/rnn/basic_lstm_cell/kernel/ApplyRMSPropApplyRMSProp#eval_net/rnn/basic_lstm_cell/kernel1train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonPtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*6
_class,
*(loc:@eval_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ctrain/RMSProp/update_eval_net/rnn/basic_lstm_cell/bias/ApplyRMSPropApplyRMSProp!eval_net/rnn/basic_lstm_cell/bias/train/eval_net/rnn/basic_lstm_cell/bias/RMSProp1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonQtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*4
_class*
(&loc:@eval_net/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
train/RMSPropNoOp4^train/RMSProp/update_eval_net/Variable/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_1/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_2/ApplyRMSProp6^train/RMSProp/update_eval_net/Variable_3/ApplyRMSPropF^train/RMSProp/update_eval_net/rnn/basic_lstm_cell/kernel/ApplyRMSPropD^train/RMSProp/update_eval_net/rnn/basic_lstm_cell/bias/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:���������)*
shape:���������)
o
target_net/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
b
target_net/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
target_net/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
-target_net/random_normal/RandomStandardNormalRandomStandardNormaltarget_net/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
target_net/random_normal/mulMul-target_net/random_normal/RandomStandardNormaltarget_net/random_normal/stddev*
T0*
_output_shapes

:

�
target_net/random_normalAddtarget_net/random_normal/multarget_net/random_normal/mean*
T0*
_output_shapes

:

�
target_net/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
target_net/Variable/AssignAssigntarget_net/Variabletarget_net/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*&
_class
loc:@target_net/Variable
�
target_net/Variable/readIdentitytarget_net/Variable*
_output_shapes

:
*
T0*&
_class
loc:@target_net/Variable
q
 target_net/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
d
target_net/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
!target_net/random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
/target_net/random_normal_1/RandomStandardNormalRandomStandardNormal target_net/random_normal_1/shape*
dtype0*
_output_shapes

:
*
seed2 *

seed *
T0
�
target_net/random_normal_1/mulMul/target_net/random_normal_1/RandomStandardNormal!target_net/random_normal_1/stddev*
_output_shapes

:
*
T0
�
target_net/random_normal_1Addtarget_net/random_normal_1/multarget_net/random_normal_1/mean*
T0*
_output_shapes

:

�
target_net/Variable_1
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
target_net/Variable_1/AssignAssigntarget_net/Variable_1target_net/random_normal_1*
T0*(
_class
loc:@target_net/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
target_net/Variable_1/readIdentitytarget_net/Variable_1*
T0*(
_class
loc:@target_net/Variable_1*
_output_shapes

:

]
target_net/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*���=
�
target_net/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes
:
*
	container *
shape:

�
target_net/Variable_2/AssignAssigntarget_net/Variable_2target_net/Const*
use_locking(*
T0*(
_class
loc:@target_net/Variable_2*
validate_shape(*
_output_shapes
:

�
target_net/Variable_2/readIdentitytarget_net/Variable_2*
T0*(
_class
loc:@target_net/Variable_2*
_output_shapes
:

_
target_net/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:
�
target_net/Variable_3
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
target_net/Variable_3/AssignAssigntarget_net/Variable_3target_net/Const_1*
use_locking(*
T0*(
_class
loc:@target_net/Variable_3*
validate_shape(*
_output_shapes
:
�
target_net/Variable_3/readIdentitytarget_net/Variable_3*
_output_shapes
:*
T0*(
_class
loc:@target_net/Variable_3
m
target_net/Reshape/shapeConst*!
valueB"����      *
dtype0*
_output_shapes
:

target_net/ReshapeReshapes_target_net/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:���������
k
target_net/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
target_net/Reshape_1Reshapetarget_net/Reshapetarget_net/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
target_net/MatMulMatMultarget_net/Reshape_1target_net/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
v
target_net/addAddtarget_net/MatMultarget_net/Variable_2/read*
T0*'
_output_shapes
:���������

o
target_net/Reshape_2/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
target_net/Reshape_2Reshapetarget_net/addtarget_net/Reshape_2/shape*
T0*
Tshape0*+
_output_shapes
:���������

q
'target_net/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
s
)target_net/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
o
-target_net/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
(target_net/BasicLSTMCellZeroState/concatConcatV2'target_net/BasicLSTMCellZeroState/Const)target_net/BasicLSTMCellZeroState/Const_1-target_net/BasicLSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
r
-target_net/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
'target_net/BasicLSTMCellZeroState/zerosFill(target_net/BasicLSTMCellZeroState/concat-target_net/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:
*
T0
s
)target_net/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
q
/target_net/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
*target_net/BasicLSTMCellZeroState/concat_1ConcatV2)target_net/BasicLSTMCellZeroState/Const_4)target_net/BasicLSTMCellZeroState/Const_5/target_net/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
t
/target_net/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
)target_net/BasicLSTMCellZeroState/zeros_1Fill*target_net/BasicLSTMCellZeroState/concat_1/target_net/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

s
)target_net/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
Q
target_net/RankConst*
value	B :*
dtype0*
_output_shapes
: 
X
target_net/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
target_net/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
target_net/rangeRangetarget_net/range/starttarget_net/Ranktarget_net/range/delta*
_output_shapes
:*

Tidx0
k
target_net/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
X
target_net/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/concatConcatV2target_net/concat/values_0target_net/rangetarget_net/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
target_net/transpose	Transposetarget_net/Reshape_2target_net/concat*
T0*+
_output_shapes
:���������
*
Tperm0
h
target_net/rnn/ShapeShapetarget_net/transpose*
_output_shapes
:*
T0*
out_type0
l
"target_net/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
n
$target_net/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
n
$target_net/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
target_net/rnn/strided_sliceStridedSlicetarget_net/rnn/Shape"target_net/rnn/strided_slice/stack$target_net/rnn/strided_slice/stack_1$target_net/rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
j
target_net/rnn/Shape_1Shapetarget_net/transpose*
T0*
out_type0*
_output_shapes
:
n
$target_net/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
target_net/rnn/strided_slice_1StridedSlicetarget_net/rnn/Shape_1$target_net/rnn/strided_slice_1/stack&target_net/rnn/strided_slice_1/stack_1&target_net/rnn/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
j
target_net/rnn/Shape_2Shapetarget_net/transpose*
T0*
out_type0*
_output_shapes
:
n
$target_net/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&target_net/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
p
&target_net/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
target_net/rnn/strided_slice_2StridedSlicetarget_net/rnn/Shape_2$target_net/rnn/strided_slice_2/stack&target_net/rnn/strided_slice_2/stack_1&target_net/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
_
target_net/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/ExpandDims
ExpandDimstarget_net/rnn/strided_slice_2target_net/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
^
target_net/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB:

\
target_net/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/concatConcatV2target_net/rnn/ExpandDimstarget_net/rnn/Consttarget_net/rnn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
_
target_net/rnn/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
target_net/rnn/zerosFilltarget_net/rnn/concattarget_net/rnn/zeros/Const*
T0*'
_output_shapes
:���������

U
target_net/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/rnn/TensorArrayTensorArrayV3target_net/rnn/strided_slice_1*:
tensor_array_name%#target_net/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
�
target_net/rnn/TensorArray_1TensorArrayV3target_net/rnn/strided_slice_1*9
tensor_array_name$"target_net/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
{
'target_net/rnn/TensorArrayUnstack/ShapeShapetarget_net/transpose*
T0*
out_type0*
_output_shapes
:

5target_net/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
7target_net/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7target_net/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/target_net/rnn/TensorArrayUnstack/strided_sliceStridedSlice'target_net/rnn/TensorArrayUnstack/Shape5target_net/rnn/TensorArrayUnstack/strided_slice/stack7target_net/rnn/TensorArrayUnstack/strided_slice/stack_17target_net/rnn/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
o
-target_net/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
o
-target_net/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
'target_net/rnn/TensorArrayUnstack/rangeRange-target_net/rnn/TensorArrayUnstack/range/start/target_net/rnn/TensorArrayUnstack/strided_slice-target_net/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Itarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3target_net/rnn/TensorArray_1'target_net/rnn/TensorArrayUnstack/rangetarget_net/transposetarget_net/rnn/TensorArray_1:1*
T0*'
_class
loc:@target_net/transpose*
_output_shapes
: 
�
target_net/rnn/while/EnterEntertarget_net/rnn/time*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context*
T0*
is_constant( 
�
target_net/rnn/while/Enter_1Entertarget_net/rnn/TensorArray:1*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context*
T0*
is_constant( 
�
target_net/rnn/while/Enter_2Enter'target_net/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*2

frame_name$"target_net/rnn/while/while_context*
T0*
is_constant( 
�
target_net/rnn/while/Enter_3Enter)target_net/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
_output_shapes

:
*2

frame_name$"target_net/rnn/while/while_context*
T0*
is_constant( 
�
target_net/rnn/while/MergeMergetarget_net/rnn/while/Enter"target_net/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
target_net/rnn/while/Merge_1Mergetarget_net/rnn/while/Enter_1$target_net/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
target_net/rnn/while/Merge_2Mergetarget_net/rnn/while/Enter_2$target_net/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
target_net/rnn/while/Merge_3Mergetarget_net/rnn/while/Enter_3$target_net/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
target_net/rnn/while/Less/EnterEntertarget_net/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context

target_net/rnn/while/LessLesstarget_net/rnn/while/Mergetarget_net/rnn/while/Less/Enter*
T0*
_output_shapes
: 
\
target_net/rnn/while/LoopCondLoopCondtarget_net/rnn/while/Less*
_output_shapes
: 
�
target_net/rnn/while/SwitchSwitchtarget_net/rnn/while/Mergetarget_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@target_net/rnn/while/Merge*
_output_shapes
: : 
�
target_net/rnn/while/Switch_1Switchtarget_net/rnn/while/Merge_1target_net/rnn/while/LoopCond*
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_1*
_output_shapes
: : 
�
target_net/rnn/while/Switch_2Switchtarget_net/rnn/while/Merge_2target_net/rnn/while/LoopCond*
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_2*(
_output_shapes
:
:

�
target_net/rnn/while/Switch_3Switchtarget_net/rnn/while/Merge_3target_net/rnn/while/LoopCond*
T0*/
_class%
#!loc:@target_net/rnn/while/Merge_3*(
_output_shapes
:
:

i
target_net/rnn/while/IdentityIdentitytarget_net/rnn/while/Switch:1*
T0*
_output_shapes
: 
m
target_net/rnn/while/Identity_1Identitytarget_net/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
u
target_net/rnn/while/Identity_2Identitytarget_net/rnn/while/Switch_2:1*
T0*
_output_shapes

:

u
target_net/rnn/while/Identity_3Identitytarget_net/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
,target_net/rnn/while/TensorArrayReadV3/EnterEntertarget_net/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*2

frame_name$"target_net/rnn/while/while_context
�
.target_net/rnn/while/TensorArrayReadV3/Enter_1EnterItarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *2

frame_name$"target_net/rnn/while/while_context
�
&target_net/rnn/while/TensorArrayReadV3TensorArrayReadV3,target_net/rnn/while/TensorArrayReadV3/Entertarget_net/rnn/while/Identity.target_net/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
Ftarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB
 *�衾
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Ntarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformFtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:(*

seed *
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
seed2 
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSubDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
Dtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulNtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
@target_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAddDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulDtarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
%target_net/rnn/basic_lstm_cell/kernel
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel
�
,target_net/rnn/basic_lstm_cell/kernel/AssignAssign%target_net/rnn/basic_lstm_cell/kernel@target_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*8
_class.
,*loc:@target_net/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
*target_net/rnn/basic_lstm_cell/kernel/readIdentity%target_net/rnn/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
�
5target_net/rnn/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:(*6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias*
valueB(*    
�
#target_net/rnn/basic_lstm_cell/bias
VariableV2*
shared_name *6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
*target_net/rnn/basic_lstm_cell/bias/AssignAssign#target_net/rnn/basic_lstm_cell/bias5target_net/rnn/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*6
_class,
*(loc:@target_net/rnn/basic_lstm_cell/bias
~
(target_net/rnn/basic_lstm_cell/bias/readIdentity#target_net/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
�
4target_net/rnn/while/rnn/basic_lstm_cell/concat/axisConst^target_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
/target_net/rnn/while/rnn/basic_lstm_cell/concatConcatV2&target_net/rnn/while/TensorArrayReadV3target_net/rnn/while/Identity_34target_net/rnn/while/rnn/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
5target_net/rnn/while/rnn/basic_lstm_cell/MatMul/EnterEnter*target_net/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*2

frame_name$"target_net/rnn/while/while_context
�
/target_net/rnn/while/rnn/basic_lstm_cell/MatMulMatMul/target_net/rnn/while/rnn/basic_lstm_cell/concat5target_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
6target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/EnterEnter(target_net/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*2

frame_name$"target_net/rnn/while/while_context
�
0target_net/rnn/while/rnn/basic_lstm_cell/BiasAddBiasAdd/target_net/rnn/while/rnn/basic_lstm_cell/MatMul6target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes

:(*
T0
�
.target_net/rnn/while/rnn/basic_lstm_cell/ConstConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
8target_net/rnn/while/rnn/basic_lstm_cell/split/split_dimConst^target_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
.target_net/rnn/while/rnn/basic_lstm_cell/splitSplit8target_net/rnn/while/rnn/basic_lstm_cell/split/split_dim0target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
.target_net/rnn/while/rnn/basic_lstm_cell/add/yConst^target_net/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
,target_net/rnn/while/rnn/basic_lstm_cell/addAdd0target_net/rnn/while/rnn/basic_lstm_cell/split:2.target_net/rnn/while/rnn/basic_lstm_cell/add/y*
T0*
_output_shapes

:

�
0target_net/rnn/while/rnn/basic_lstm_cell/SigmoidSigmoid,target_net/rnn/while/rnn/basic_lstm_cell/add*
_output_shapes

:
*
T0
�
,target_net/rnn/while/rnn/basic_lstm_cell/mulMultarget_net/rnn/while/Identity_20target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1Sigmoid.target_net/rnn/while/rnn/basic_lstm_cell/split*
T0*
_output_shapes

:

�
-target_net/rnn/while/rnn/basic_lstm_cell/TanhTanh0target_net/rnn/while/rnn/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/mul_1Mul2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1-target_net/rnn/while/rnn/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/add_1Add,target_net/rnn/while/rnn/basic_lstm_cell/mul.target_net/rnn/while/rnn/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
/target_net/rnn/while/rnn/basic_lstm_cell/Tanh_1Tanh.target_net/rnn/while/rnn/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2Sigmoid0target_net/rnn/while/rnn/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
.target_net/rnn/while/rnn/basic_lstm_cell/mul_2Mul/target_net/rnn/while/rnn/basic_lstm_cell/Tanh_12target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2*
_output_shapes

:
*
T0
�
>target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntertarget_net/rnn/TensorArray*
T0*A
_class7
53loc:@target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*2

frame_name$"target_net/rnn/while/while_context
�
8target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3>target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entertarget_net/rnn/while/Identity.target_net/rnn/while/rnn/basic_lstm_cell/mul_2target_net/rnn/while/Identity_1*
T0*A
_class7
53loc:@target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
_output_shapes
: 
|
target_net/rnn/while/add/yConst^target_net/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
{
target_net/rnn/while/addAddtarget_net/rnn/while/Identitytarget_net/rnn/while/add/y*
_output_shapes
: *
T0
n
"target_net/rnn/while/NextIterationNextIterationtarget_net/rnn/while/add*
T0*
_output_shapes
: 
�
$target_net/rnn/while/NextIteration_1NextIteration8target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
$target_net/rnn/while/NextIteration_2NextIteration.target_net/rnn/while/rnn/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
$target_net/rnn/while/NextIteration_3NextIteration.target_net/rnn/while/rnn/basic_lstm_cell/mul_2*
_output_shapes

:
*
T0
_
target_net/rnn/while/ExitExittarget_net/rnn/while/Switch*
T0*
_output_shapes
: 
c
target_net/rnn/while/Exit_1Exittarget_net/rnn/while/Switch_1*
T0*
_output_shapes
: 
k
target_net/rnn/while/Exit_2Exittarget_net/rnn/while/Switch_2*
_output_shapes

:
*
T0
k
target_net/rnn/while/Exit_3Exittarget_net/rnn/while/Switch_3*
_output_shapes

:
*
T0
�
1target_net/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3target_net/rnn/TensorArraytarget_net/rnn/while/Exit_1*-
_class#
!loc:@target_net/rnn/TensorArray*
_output_shapes
: 
�
+target_net/rnn/TensorArrayStack/range/startConst*
value	B : *-
_class#
!loc:@target_net/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
+target_net/rnn/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@target_net/rnn/TensorArray
�
%target_net/rnn/TensorArrayStack/rangeRange+target_net/rnn/TensorArrayStack/range/start1target_net/rnn/TensorArrayStack/TensorArraySizeV3+target_net/rnn/TensorArrayStack/range/delta*-
_class#
!loc:@target_net/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
3target_net/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3target_net/rnn/TensorArray%target_net/rnn/TensorArrayStack/rangetarget_net/rnn/while/Exit_1*
dtype0*"
_output_shapes
:
*
element_shape
:
*-
_class#
!loc:@target_net/rnn/TensorArray
`
target_net/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
U
target_net/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
\
target_net/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
\
target_net/rnn/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
target_net/rnn/rangeRangetarget_net/rnn/range/starttarget_net/rnn/Ranktarget_net/rnn/range/delta*
_output_shapes
:*

Tidx0
q
 target_net/rnn/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
^
target_net/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
target_net/rnn/concat_1ConcatV2 target_net/rnn/concat_1/values_0target_net/rnn/rangetarget_net/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
target_net/rnn/transpose	Transpose3target_net/rnn/TensorArrayStack/TensorArrayGatherV3target_net/rnn/concat_1*
Tperm0*
T0*"
_output_shapes
:

�
target_net/MatMul_1MatMultarget_net/rnn/while/Exit_3target_net/Variable_1/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
q
target_net/add_1Addtarget_net/MatMul_1target_net/Variable_3/read*
T0*
_output_shapes

:
�
AssignAssigntarget_net/Variableeval_net/Variable/read*
use_locking(*
T0*&
_class
loc:@target_net/Variable*
validate_shape(*
_output_shapes

:

�
Assign_1Assigntarget_net/Variable_1eval_net/Variable_1/read*
use_locking(*
T0*(
_class
loc:@target_net/Variable_1*
validate_shape(*
_output_shapes

:

�
Assign_2Assigntarget_net/Variable_2eval_net/Variable_2/read*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*(
_class
loc:@target_net/Variable_2
�
Assign_3Assigntarget_net/Variable_3eval_net/Variable_3/read*
use_locking(*
T0*(
_class
loc:@target_net/Variable_3*
validate_shape(*
_output_shapes
:""
train_op

train/RMSProp"o
eval_net_params\
Z
eval_net/Variable:0
eval_net/Variable_1:0
eval_net/Variable_2:0
eval_net/Variable_3:0"�
trainable_variables��
c
eval_net/Variable:0eval_net/Variable/Assigneval_net/Variable/read:02eval_net/random_normal:0
k
eval_net/Variable_1:0eval_net/Variable_1/Assigneval_net/Variable_1/read:02eval_net/random_normal_1:0
a
eval_net/Variable_2:0eval_net/Variable_2/Assigneval_net/Variable_2/read:02eval_net/Const:0
c
eval_net/Variable_3:0eval_net/Variable_3/Assigneval_net/Variable_3/read:02eval_net/Const_1:0
�
%eval_net/rnn/basic_lstm_cell/kernel:0*eval_net/rnn/basic_lstm_cell/kernel/Assign*eval_net/rnn/basic_lstm_cell/kernel/read:02@eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
#eval_net/rnn/basic_lstm_cell/bias:0(eval_net/rnn/basic_lstm_cell/bias/Assign(eval_net/rnn/basic_lstm_cell/bias/read:025eval_net/rnn/basic_lstm_cell/bias/Initializer/Const:0
k
target_net/Variable:0target_net/Variable/Assigntarget_net/Variable/read:02target_net/random_normal:0
s
target_net/Variable_1:0target_net/Variable_1/Assigntarget_net/Variable_1/read:02target_net/random_normal_1:0
i
target_net/Variable_2:0target_net/Variable_2/Assigntarget_net/Variable_2/read:02target_net/Const:0
k
target_net/Variable_3:0target_net/Variable_3/Assigntarget_net/Variable_3/read:02target_net/Const_1:0
�
'target_net/rnn/basic_lstm_cell/kernel:0,target_net/rnn/basic_lstm_cell/kernel/Assign,target_net/rnn/basic_lstm_cell/kernel/read:02Btarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
%target_net/rnn/basic_lstm_cell/bias:0*target_net/rnn/basic_lstm_cell/bias/Assign*target_net/rnn/basic_lstm_cell/bias/read:027target_net/rnn/basic_lstm_cell/bias/Initializer/Const:0"y
target_net_paramsd
b
target_net/Variable:0
target_net/Variable_1:0
target_net/Variable_2:0
target_net/Variable_3:0"�
	variables��
c
eval_net/Variable:0eval_net/Variable/Assigneval_net/Variable/read:02eval_net/random_normal:0
k
eval_net/Variable_1:0eval_net/Variable_1/Assigneval_net/Variable_1/read:02eval_net/random_normal_1:0
a
eval_net/Variable_2:0eval_net/Variable_2/Assigneval_net/Variable_2/read:02eval_net/Const:0
c
eval_net/Variable_3:0eval_net/Variable_3/Assigneval_net/Variable_3/read:02eval_net/Const_1:0
�
%eval_net/rnn/basic_lstm_cell/kernel:0*eval_net/rnn/basic_lstm_cell/kernel/Assign*eval_net/rnn/basic_lstm_cell/kernel/read:02@eval_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
#eval_net/rnn/basic_lstm_cell/bias:0(eval_net/rnn/basic_lstm_cell/bias/Assign(eval_net/rnn/basic_lstm_cell/bias/read:025eval_net/rnn/basic_lstm_cell/bias/Initializer/Const:0
�
!train/eval_net/Variable/RMSProp:0&train/eval_net/Variable/RMSProp/Assign&train/eval_net/Variable/RMSProp/read:022train/eval_net/Variable/RMSProp/Initializer/ones:0
�
#train/eval_net/Variable/RMSProp_1:0(train/eval_net/Variable/RMSProp_1/Assign(train/eval_net/Variable/RMSProp_1/read:025train/eval_net/Variable/RMSProp_1/Initializer/zeros:0
�
#train/eval_net/Variable_1/RMSProp:0(train/eval_net/Variable_1/RMSProp/Assign(train/eval_net/Variable_1/RMSProp/read:024train/eval_net/Variable_1/RMSProp/Initializer/ones:0
�
%train/eval_net/Variable_1/RMSProp_1:0*train/eval_net/Variable_1/RMSProp_1/Assign*train/eval_net/Variable_1/RMSProp_1/read:027train/eval_net/Variable_1/RMSProp_1/Initializer/zeros:0
�
#train/eval_net/Variable_2/RMSProp:0(train/eval_net/Variable_2/RMSProp/Assign(train/eval_net/Variable_2/RMSProp/read:024train/eval_net/Variable_2/RMSProp/Initializer/ones:0
�
%train/eval_net/Variable_2/RMSProp_1:0*train/eval_net/Variable_2/RMSProp_1/Assign*train/eval_net/Variable_2/RMSProp_1/read:027train/eval_net/Variable_2/RMSProp_1/Initializer/zeros:0
�
#train/eval_net/Variable_3/RMSProp:0(train/eval_net/Variable_3/RMSProp/Assign(train/eval_net/Variable_3/RMSProp/read:024train/eval_net/Variable_3/RMSProp/Initializer/ones:0
�
%train/eval_net/Variable_3/RMSProp_1:0*train/eval_net/Variable_3/RMSProp_1/Assign*train/eval_net/Variable_3/RMSProp_1/read:027train/eval_net/Variable_3/RMSProp_1/Initializer/zeros:0
�
3train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp:08train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Assign8train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/read:02Dtrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp/Initializer/ones:0
�
5train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1:0:train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Assign:train/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/read:02Gtrain/eval_net/rnn/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros:0
�
1train/eval_net/rnn/basic_lstm_cell/bias/RMSProp:06train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Assign6train/eval_net/rnn/basic_lstm_cell/bias/RMSProp/read:02Btrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp/Initializer/ones:0
�
3train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1:08train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Assign8train/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/read:02Etrain/eval_net/rnn/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros:0
k
target_net/Variable:0target_net/Variable/Assigntarget_net/Variable/read:02target_net/random_normal:0
s
target_net/Variable_1:0target_net/Variable_1/Assigntarget_net/Variable_1/read:02target_net/random_normal_1:0
i
target_net/Variable_2:0target_net/Variable_2/Assigntarget_net/Variable_2/read:02target_net/Const:0
k
target_net/Variable_3:0target_net/Variable_3/Assigntarget_net/Variable_3/read:02target_net/Const_1:0
�
'target_net/rnn/basic_lstm_cell/kernel:0,target_net/rnn/basic_lstm_cell/kernel/Assign,target_net/rnn/basic_lstm_cell/kernel/read:02Btarget_net/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
%target_net/rnn/basic_lstm_cell/bias:0*target_net/rnn/basic_lstm_cell/bias/Assign*target_net/rnn/basic_lstm_cell/bias/read:027target_net/rnn/basic_lstm_cell/bias/Initializer/Const:0"�`
while_context�_�_
�A
 eval_net/rnn/while/while_context *eval_net/rnn/while/LoopCond:02eval_net/rnn/while/Merge:0:eval_net/rnn/while/Identity:0Beval_net/rnn/while/Exit:0Beval_net/rnn/while/Exit_1:0Beval_net/rnn/while/Exit_2:0Beval_net/rnn/while/Exit_3:0Btrain/gradients/f_count_2:0J�>
eval_net/rnn/TensorArray:0
Ieval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
eval_net/rnn/TensorArray_1:0
(eval_net/rnn/basic_lstm_cell/bias/read:0
*eval_net/rnn/basic_lstm_cell/kernel/read:0
eval_net/rnn/strided_slice_1:0
eval_net/rnn/while/Enter:0
eval_net/rnn/while/Enter_1:0
eval_net/rnn/while/Enter_2:0
eval_net/rnn/while/Enter_3:0
eval_net/rnn/while/Exit:0
eval_net/rnn/while/Exit_1:0
eval_net/rnn/while/Exit_2:0
eval_net/rnn/while/Exit_3:0
eval_net/rnn/while/Identity:0
eval_net/rnn/while/Identity_1:0
eval_net/rnn/while/Identity_2:0
eval_net/rnn/while/Identity_3:0
eval_net/rnn/while/Less/Enter:0
eval_net/rnn/while/Less:0
eval_net/rnn/while/LoopCond:0
eval_net/rnn/while/Merge:0
eval_net/rnn/while/Merge:1
eval_net/rnn/while/Merge_1:0
eval_net/rnn/while/Merge_1:1
eval_net/rnn/while/Merge_2:0
eval_net/rnn/while/Merge_2:1
eval_net/rnn/while/Merge_3:0
eval_net/rnn/while/Merge_3:1
"eval_net/rnn/while/NextIteration:0
$eval_net/rnn/while/NextIteration_1:0
$eval_net/rnn/while/NextIteration_2:0
$eval_net/rnn/while/NextIteration_3:0
eval_net/rnn/while/Switch:0
eval_net/rnn/while/Switch:1
eval_net/rnn/while/Switch_1:0
eval_net/rnn/while/Switch_1:1
eval_net/rnn/while/Switch_2:0
eval_net/rnn/while/Switch_2:1
eval_net/rnn/while/Switch_3:0
eval_net/rnn/while/Switch_3:1
,eval_net/rnn/while/TensorArrayReadV3/Enter:0
.eval_net/rnn/while/TensorArrayReadV3/Enter_1:0
&eval_net/rnn/while/TensorArrayReadV3:0
>eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
8eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
eval_net/rnn/while/add/y:0
eval_net/rnn/while/add:0
6eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter:0
0eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd:0
.eval_net/rnn/while/rnn/basic_lstm_cell/Const:0
5eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter:0
/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul:0
0eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid:0
2eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1:0
2eval_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2:0
-eval_net/rnn/while/rnn/basic_lstm_cell/Tanh:0
/eval_net/rnn/while/rnn/basic_lstm_cell/Tanh_1:0
.eval_net/rnn/while/rnn/basic_lstm_cell/add/y:0
,eval_net/rnn/while/rnn/basic_lstm_cell/add:0
.eval_net/rnn/while/rnn/basic_lstm_cell/add_1:0
4eval_net/rnn/while/rnn/basic_lstm_cell/concat/axis:0
/eval_net/rnn/while/rnn/basic_lstm_cell/concat:0
,eval_net/rnn/while/rnn/basic_lstm_cell/mul:0
.eval_net/rnn/while/rnn/basic_lstm_cell/mul_1:0
.eval_net/rnn/while/rnn/basic_lstm_cell/mul_2:0
8eval_net/rnn/while/rnn/basic_lstm_cell/split/split_dim:0
.eval_net/rnn/while/rnn/basic_lstm_cell/split:0
.eval_net/rnn/while/rnn/basic_lstm_cell/split:1
.eval_net/rnn/while/rnn/basic_lstm_cell/split:2
.eval_net/rnn/while/rnn/basic_lstm_cell/split:3
train/gradients/Add/y:0
train/gradients/Add:0
train/gradients/Merge:0
train/gradients/Merge:1
train/gradients/NextIteration:0
train/gradients/Switch:0
train/gradients/Switch:1
etrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Enter:0
ktrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2:0
etrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Jtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/Shape:0
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
Wtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
Ytrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/Enter:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/StackPushV2:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc:0
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/Enter:0
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2:0
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/Enter:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/StackPushV2:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc:0
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/Enter:0
Utrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2:0
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/Enter:0
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/StackPushV2:0
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/Enter:0
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/StackPushV2:0
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc:0
train/gradients/f_count:0
train/gradients/f_count_1:0
train/gradients/f_count_2:0{
Ieval_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0.eval_net/rnn/while/TensorArrayReadV3/Enter_1:0�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
etrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0etrain/gradients/eval_net/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Enter:0b
(eval_net/rnn/basic_lstm_cell/bias/read:06eval_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter:0�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/f_acc:0Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul_1/Enter:0�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/f_acc:0Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul/Enter:0\
eval_net/rnn/TensorArray:0>eval_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/f_acc:0Mtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul/Enter:0�
Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc:0Qtrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter:0�
Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/f_acc:0Ktrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_grad/mul/Enter:0L
eval_net/rnn/TensorArray_1:0,eval_net/rnn/while/TensorArrayReadV3/Enter:0A
eval_net/rnn/strided_slice_1:0eval_net/rnn/while/Less/Enter:0�
Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Strain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_2_grad/mul_1/Enter:0c
*eval_net/rnn/basic_lstm_cell/kernel/read:05eval_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter:0�
Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0Otrain/gradients/eval_net/rnn/while/rnn/basic_lstm_cell/mul_1_grad/mul_1/Enter:0Reval_net/rnn/while/Enter:0Reval_net/rnn/while/Enter_1:0Reval_net/rnn/while/Enter_2:0Reval_net/rnn/while/Enter_3:0Rtrain/gradients/f_count_1:0
�
"target_net/rnn/while/while_context *target_net/rnn/while/LoopCond:02target_net/rnn/while/Merge:0:target_net/rnn/while/Identity:0Btarget_net/rnn/while/Exit:0Btarget_net/rnn/while/Exit_1:0Btarget_net/rnn/while/Exit_2:0Btarget_net/rnn/while/Exit_3:0J�
target_net/rnn/TensorArray:0
Ktarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
target_net/rnn/TensorArray_1:0
*target_net/rnn/basic_lstm_cell/bias/read:0
,target_net/rnn/basic_lstm_cell/kernel/read:0
 target_net/rnn/strided_slice_1:0
target_net/rnn/while/Enter:0
target_net/rnn/while/Enter_1:0
target_net/rnn/while/Enter_2:0
target_net/rnn/while/Enter_3:0
target_net/rnn/while/Exit:0
target_net/rnn/while/Exit_1:0
target_net/rnn/while/Exit_2:0
target_net/rnn/while/Exit_3:0
target_net/rnn/while/Identity:0
!target_net/rnn/while/Identity_1:0
!target_net/rnn/while/Identity_2:0
!target_net/rnn/while/Identity_3:0
!target_net/rnn/while/Less/Enter:0
target_net/rnn/while/Less:0
target_net/rnn/while/LoopCond:0
target_net/rnn/while/Merge:0
target_net/rnn/while/Merge:1
target_net/rnn/while/Merge_1:0
target_net/rnn/while/Merge_1:1
target_net/rnn/while/Merge_2:0
target_net/rnn/while/Merge_2:1
target_net/rnn/while/Merge_3:0
target_net/rnn/while/Merge_3:1
$target_net/rnn/while/NextIteration:0
&target_net/rnn/while/NextIteration_1:0
&target_net/rnn/while/NextIteration_2:0
&target_net/rnn/while/NextIteration_3:0
target_net/rnn/while/Switch:0
target_net/rnn/while/Switch:1
target_net/rnn/while/Switch_1:0
target_net/rnn/while/Switch_1:1
target_net/rnn/while/Switch_2:0
target_net/rnn/while/Switch_2:1
target_net/rnn/while/Switch_3:0
target_net/rnn/while/Switch_3:1
.target_net/rnn/while/TensorArrayReadV3/Enter:0
0target_net/rnn/while/TensorArrayReadV3/Enter_1:0
(target_net/rnn/while/TensorArrayReadV3:0
@target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
:target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
target_net/rnn/while/add/y:0
target_net/rnn/while/add:0
8target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter:0
2target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd:0
0target_net/rnn/while/rnn/basic_lstm_cell/Const:0
7target_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter:0
1target_net/rnn/while/rnn/basic_lstm_cell/MatMul:0
2target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid:0
4target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_1:0
4target_net/rnn/while/rnn/basic_lstm_cell/Sigmoid_2:0
/target_net/rnn/while/rnn/basic_lstm_cell/Tanh:0
1target_net/rnn/while/rnn/basic_lstm_cell/Tanh_1:0
0target_net/rnn/while/rnn/basic_lstm_cell/add/y:0
.target_net/rnn/while/rnn/basic_lstm_cell/add:0
0target_net/rnn/while/rnn/basic_lstm_cell/add_1:0
6target_net/rnn/while/rnn/basic_lstm_cell/concat/axis:0
1target_net/rnn/while/rnn/basic_lstm_cell/concat:0
.target_net/rnn/while/rnn/basic_lstm_cell/mul:0
0target_net/rnn/while/rnn/basic_lstm_cell/mul_1:0
0target_net/rnn/while/rnn/basic_lstm_cell/mul_2:0
:target_net/rnn/while/rnn/basic_lstm_cell/split/split_dim:0
0target_net/rnn/while/rnn/basic_lstm_cell/split:0
0target_net/rnn/while/rnn/basic_lstm_cell/split:1
0target_net/rnn/while/rnn/basic_lstm_cell/split:2
0target_net/rnn/while/rnn/basic_lstm_cell/split:3E
 target_net/rnn/strided_slice_1:0!target_net/rnn/while/Less/Enter:0
Ktarget_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:00target_net/rnn/while/TensorArrayReadV3/Enter_1:0`
target_net/rnn/TensorArray:0@target_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0g
,target_net/rnn/basic_lstm_cell/kernel/read:07target_net/rnn/while/rnn/basic_lstm_cell/MatMul/Enter:0f
*target_net/rnn/basic_lstm_cell/bias/read:08target_net/rnn/while/rnn/basic_lstm_cell/BiasAdd/Enter:0P
target_net/rnn/TensorArray_1:0.target_net/rnn/while/TensorArrayReadV3/Enter:0Rtarget_net/rnn/while/Enter:0Rtarget_net/rnn/while/Enter_1:0Rtarget_net/rnn/while/Enter_2:0Rtarget_net/rnn/while/Enter_3:0.�/U