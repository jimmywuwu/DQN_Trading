       �K"	   `���Abrain.Event:25�h���     `���	&�`���A"ۃ
l
sPlaceholder* 
shape:���������
*
dtype0*+
_output_shapes
:���������

k
Q_targetPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
i
$eval_net/l1/DropoutWrapperInit/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
&eval_net/l1/DropoutWrapperInit/Const_1Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
&eval_net/l1/DropoutWrapperInit/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Weval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstXeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillWeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:

�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
�
^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Yeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillYeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:
*
T0
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
R
eval_net/l1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Y
eval_net/l1/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
eval_net/l1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_net/l1/rangeRangeeval_net/l1/range/starteval_net/l1/Rankeval_net/l1/range/delta*
_output_shapes
:*

Tidx0
l
eval_net/l1/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
eval_net/l1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/l1/concatConcatV2eval_net/l1/concat/values_0eval_net/l1/rangeeval_net/l1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
|
eval_net/l1/transpose	Transposeseval_net/l1/concat*+
_output_shapes
:
���������*
Tperm0*
T0
j
eval_net/l1/rnn/ShapeShapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
m
#eval_net/l1/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%eval_net/l1/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%eval_net/l1/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
eval_net/l1/rnn/strided_sliceStridedSliceeval_net/l1/rnn/Shape#eval_net/l1/rnn/strided_slice/stack%eval_net/l1/rnn/strided_slice/stack_1%eval_net/l1/rnn/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
l
eval_net/l1/rnn/Shape_1Shapeeval_net/l1/transpose*
_output_shapes
:*
T0*
out_type0
o
%eval_net/l1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/l1/rnn/strided_slice_1StridedSliceeval_net/l1/rnn/Shape_1%eval_net/l1/rnn/strided_slice_1/stack'eval_net/l1/rnn/strided_slice_1/stack_1'eval_net/l1/rnn/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
l
eval_net/l1/rnn/Shape_2Shapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
o
%eval_net/l1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/l1/rnn/strided_slice_2StridedSliceeval_net/l1/rnn/Shape_2%eval_net/l1/rnn/strided_slice_2/stack'eval_net/l1/rnn/strided_slice_2/stack_1'eval_net/l1/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
`
eval_net/l1/rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/l1/rnn/ExpandDims
ExpandDimseval_net/l1/rnn/strided_slice_2eval_net/l1/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
_
eval_net/l1/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
]
eval_net/l1/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/concatConcatV2eval_net/l1/rnn/ExpandDimseval_net/l1/rnn/Consteval_net/l1/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
`
eval_net/l1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/zerosFilleval_net/l1/rnn/concateval_net/l1/rnn/zeros/Const*
T0*'
_output_shapes
:���������

V
eval_net/l1/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/l1/rnn/TensorArrayTensorArrayV3eval_net/l1/rnn/strided_slice_1*
element_shape:*
dynamic_size( *
clear_after_read(*;
tensor_array_name&$eval_net/l1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
eval_net/l1/rnn/TensorArray_1TensorArrayV3eval_net/l1/rnn/strided_slice_1*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*:
tensor_array_name%#eval_net/l1/rnn/dynamic_rnn/input_0
}
(eval_net/l1/rnn/TensorArrayUnstack/ShapeShapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
�
6eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
0eval_net/l1/rnn/TensorArrayUnstack/strided_sliceStridedSlice(eval_net/l1/rnn/TensorArrayUnstack/Shape6eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_18eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
p
.eval_net/l1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
p
.eval_net/l1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
(eval_net/l1/rnn/TensorArrayUnstack/rangeRange.eval_net/l1/rnn/TensorArrayUnstack/range/start0eval_net/l1/rnn/TensorArrayUnstack/strided_slice.eval_net/l1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Jeval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3eval_net/l1/rnn/TensorArray_1(eval_net/l1/rnn/TensorArrayUnstack/rangeeval_net/l1/transposeeval_net/l1/rnn/TensorArray_1:1*
T0*(
_class
loc:@eval_net/l1/transpose*
_output_shapes
: 
�
eval_net/l1/rnn/while/EnterEntereval_net/l1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/Enter_1Entereval_net/l1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/Enter_2EnterVeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
eval_net/l1/rnn/while/Enter_3EnterXeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/MergeMergeeval_net/l1/rnn/while/Enter#eval_net/l1/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
eval_net/l1/rnn/while/Merge_1Mergeeval_net/l1/rnn/while/Enter_1%eval_net/l1/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
eval_net/l1/rnn/while/Merge_2Mergeeval_net/l1/rnn/while/Enter_2%eval_net/l1/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
eval_net/l1/rnn/while/Merge_3Mergeeval_net/l1/rnn/while/Enter_3%eval_net/l1/rnn/while/NextIteration_3*
N* 
_output_shapes
:
: *
T0
�
 eval_net/l1/rnn/while/Less/EnterEntereval_net/l1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/LessLesseval_net/l1/rnn/while/Merge eval_net/l1/rnn/while/Less/Enter*
_output_shapes
: *
T0
^
eval_net/l1/rnn/while/LoopCondLoopCondeval_net/l1/rnn/while/Less*
_output_shapes
: 
�
eval_net/l1/rnn/while/SwitchSwitcheval_net/l1/rnn/while/Mergeeval_net/l1/rnn/while/LoopCond*
T0*.
_class$
" loc:@eval_net/l1/rnn/while/Merge*
_output_shapes
: : 
�
eval_net/l1/rnn/while/Switch_1Switcheval_net/l1/rnn/while/Merge_1eval_net/l1/rnn/while/LoopCond*
_output_shapes
: : *
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_1
�
eval_net/l1/rnn/while/Switch_2Switcheval_net/l1/rnn/while/Merge_2eval_net/l1/rnn/while/LoopCond*(
_output_shapes
:
:
*
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_2
�
eval_net/l1/rnn/while/Switch_3Switcheval_net/l1/rnn/while/Merge_3eval_net/l1/rnn/while/LoopCond*
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_3*(
_output_shapes
:
:

k
eval_net/l1/rnn/while/IdentityIdentityeval_net/l1/rnn/while/Switch:1*
T0*
_output_shapes
: 
o
 eval_net/l1/rnn/while/Identity_1Identity eval_net/l1/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
w
 eval_net/l1/rnn/while/Identity_2Identity eval_net/l1/rnn/while/Switch_2:1*
T0*
_output_shapes

:

w
 eval_net/l1/rnn/while/Identity_3Identity eval_net/l1/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
-eval_net/l1/rnn/while/TensorArrayReadV3/EnterEntereval_net/l1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
/eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1EnterJeval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
'eval_net/l1/rnn/while/TensorArrayReadV3TensorArrayReadV3-eval_net/l1/rnn/while/TensorArrayReadV3/Entereval_net/l1/rnn/while/Identity/eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
]eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D��
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
eeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform]eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:(*

seed *
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSub[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/max[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMuleeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Weval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAdd[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mul[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelWeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
Leval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:(*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    
�
:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(
�
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasLeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
use_locking(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
?eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(*
T0
�
Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2'eval_net/l1/rnn/while/TensorArrayReadV3 eval_net/l1/rnn/while/Identity_3Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
Seval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnterAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*3

frame_name%#eval_net/l1/rnn/while/while_context
�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatSeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
Teval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter?eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*3

frame_name%#eval_net/l1/rnn/while/while_context
�
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulTeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Veval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitVeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*<
_output_shapes*
(:
:
:
:
*
	num_split*
T0
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^eval_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
T0*
_output_shapes

:

�
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoidJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0*
_output_shapes

:

�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMul eval_net/l1/rnn/while/Identity_2Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:
*
T0
�
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1SigmoidLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0*
_output_shapes

:

�
Keval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Keval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1AddJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1TanhLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
_output_shapes

:
*
T0
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
Aeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probConst^eval_net/l1/rnn/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^eval_net/l1/rnn/while/Identity*
valueB"   
   *
dtype0*
_output_shapes
:
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^eval_net/l1/rnn/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^eval_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Teval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSubJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulTeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
_output_shapes

:
*
T0
�
Feval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAddJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes

:

�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAddAeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probFeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
T0*
_output_shapes

:

�
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
_output_shapes

:
*
T0
�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDivLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2Aeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob*
T0*
_output_shapes

:

�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
_output_shapes

:

�
?eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntereval_net/l1/rnn/TensorArray*
is_constant(*
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations 
�
9eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3?eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/l1/rnn/while/Identity;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul eval_net/l1/rnn/while/Identity_1*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes
: 
~
eval_net/l1/rnn/while/add/yConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
~
eval_net/l1/rnn/while/addAddeval_net/l1/rnn/while/Identityeval_net/l1/rnn/while/add/y*
T0*
_output_shapes
: 
p
#eval_net/l1/rnn/while/NextIterationNextIterationeval_net/l1/rnn/while/add*
T0*
_output_shapes
: 
�
%eval_net/l1/rnn/while/NextIteration_1NextIteration9eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
%eval_net/l1/rnn/while/NextIteration_2NextIterationLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
%eval_net/l1/rnn/while/NextIteration_3NextIterationLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
T0*
_output_shapes

:

a
eval_net/l1/rnn/while/ExitExiteval_net/l1/rnn/while/Switch*
T0*
_output_shapes
: 
e
eval_net/l1/rnn/while/Exit_1Exiteval_net/l1/rnn/while/Switch_1*
T0*
_output_shapes
: 
m
eval_net/l1/rnn/while/Exit_2Exiteval_net/l1/rnn/while/Switch_2*
T0*
_output_shapes

:

m
eval_net/l1/rnn/while/Exit_3Exiteval_net/l1/rnn/while/Switch_3*
T0*
_output_shapes

:

�
2eval_net/l1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3eval_net/l1/rnn/TensorArrayeval_net/l1/rnn/while/Exit_1*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
_output_shapes
: 
�
,eval_net/l1/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *.
_class$
" loc:@eval_net/l1/rnn/TensorArray
�
,eval_net/l1/rnn/TensorArrayStack/range/deltaConst*
value	B :*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
&eval_net/l1/rnn/TensorArrayStack/rangeRange,eval_net/l1/rnn/TensorArrayStack/range/start2eval_net/l1/rnn/TensorArrayStack/TensorArraySizeV3,eval_net/l1/rnn/TensorArrayStack/range/delta*#
_output_shapes
:���������*

Tidx0*.
_class$
" loc:@eval_net/l1/rnn/TensorArray
�
4eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3eval_net/l1/rnn/TensorArray&eval_net/l1/rnn/TensorArrayStack/rangeeval_net/l1/rnn/while/Exit_1*
element_shape
:
*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
dtype0*"
_output_shapes
:


a
eval_net/l1/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
V
eval_net/l1/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
]
eval_net/l1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
]
eval_net/l1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/rangeRangeeval_net/l1/rnn/range/starteval_net/l1/rnn/Rankeval_net/l1/rnn/range/delta*
_output_shapes
:*

Tidx0
r
!eval_net/l1/rnn/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
_
eval_net/l1/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/concat_1ConcatV2!eval_net/l1/rnn/concat_1/values_0eval_net/l1/rnn/rangeeval_net/l1/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
eval_net/l1/rnn/transpose	Transpose4eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3eval_net/l1/rnn/concat_1*
T0*"
_output_shapes
:

*
Tperm0
t
eval_net/l1/strided_slice/stackConst*!
valueB"    ����    *
dtype0*
_output_shapes
:
v
!eval_net/l1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
v
!eval_net/l1/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
eval_net/l1/strided_sliceStridedSliceeval_net/l1/rnn/transposeeval_net/l1/strided_slice/stack!eval_net/l1/strided_slice/stack_1!eval_net/l1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes

:
*
Index0*
T0
�
.eval_net/l2/w2/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@eval_net/l2/w2*
valueB"
      
�
-eval_net/l2/w2/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/eval_net/l2/w2/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l2/w2/Initializer/random_normal/shape*
T0*!
_class
loc:@eval_net/l2/w2*
seed2 *
dtype0*
_output_shapes

:
*

seed 
�
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
eval_net/l2/w2
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l2/w2*
	container *
shape
:

�
eval_net/l2/w2/AssignAssigneval_net/l2/w2(eval_net/l2/w2/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

{
eval_net/l2/w2/readIdentityeval_net/l2/w2*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
 eval_net/l2/b2/Initializer/ConstConst*!
_class
loc:@eval_net/l2/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
eval_net/l2/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container *
shape
:
�
eval_net/l2/b2/AssignAssigneval_net/l2/b2 eval_net/l2/b2/Initializer/Const*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
{
eval_net/l2/b2/readIdentityeval_net/l2/b2*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
�
eval_net/l2/MatMulMatMuleval_net/l1/strided_sliceeval_net/l2/w2/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
h
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*
T0*
_output_shapes

:
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*
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
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
z
train/gradients/SwitchSwitchtrain/gradients/Mergeeval_net/l1/rnn/while/LoopCond*
T0*
_output_shapes
: : 
x
train/gradients/Add/yConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
_output_shapes
: *
T0
�
train/gradients/NextIterationNextIterationtrain/gradients/Addm^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2c^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2a^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2p^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2x^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2v^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2x^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
_output_shapes
: *
T0
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
T0*
_output_shapes
: 
Y
train/gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
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
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
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
train/gradients/NextIteration_1NextIterationtrain/gradients/Subh^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
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
valueB"      *
dtype0*
_output_shapes
:
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
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
�
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
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
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
_output_shapes

:*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg
{
*train/gradients/eval_net/l2/add_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*
_output_shapes

:*
T0*
Tshape0
�
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
�
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape*
_output_shapes

:
�
?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1*
_output_shapes

:
�
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*
_output_shapes

:
*
transpose_a( *
transpose_b(*
T0
�
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/strided_slice=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
�
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
�
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul*
_output_shapes

:

�
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1*
_output_shapes

:

�
4train/gradients/eval_net/l1/strided_slice_grad/ShapeConst*!
valueB"   
   
   *
dtype0*
_output_shapes
:
�
?train/gradients/eval_net/l1/strided_slice_grad/StridedSliceGradStridedSliceGrad4train/gradients/eval_net/l1/strided_slice_grad/Shapeeval_net/l1/strided_slice/stack!eval_net/l1/strided_slice/stack_1!eval_net/l1/strided_slice/stack_2@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*"
_output_shapes
:


�
@train/gradients/eval_net/l1/rnn/transpose_grad/InvertPermutationInvertPermutationeval_net/l1/rnn/concat_1*
T0*
_output_shapes
:
�
8train/gradients/eval_net/l1/rnn/transpose_grad/transpose	Transpose?train/gradients/eval_net/l1/strided_slice_grad/StridedSliceGrad@train/gradients/eval_net/l1/rnn/transpose_grad/InvertPermutation*
T0*"
_output_shapes
:

*
Tperm0
�
ktrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3eval_net/l1/rnn/TensorArrayeval_net/l1/rnn/while/Exit_1*
_output_shapes

:: *
sourcetrain/gradients*.
_class$
" loc:@eval_net/l1/rnn/TensorArray
�
gtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityeval_net/l1/rnn/while/Exit_1l^train/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
_output_shapes
: 
�
qtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ktrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3&eval_net/l1/rnn/TensorArrayStack/range8train/gradients/eval_net/l1/rnn/transpose_grad/transposegtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
train/gradients/zerosConst*
dtype0*
_output_shapes

:
*
valueB
*    
l
train/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
8train/gradients/eval_net/l1/rnn/while/Exit_1_grad/b_exitEnterqtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
8train/gradients/eval_net/l1/rnn/while/Exit_2_grad/b_exitEntertrain/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
8train/gradients/eval_net/l1/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
<train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_1_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_1_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
<train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_2_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
<train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_3_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_3_grad_1/NextIteration*
N* 
_output_shapes
:
: *
T0
�
9train/gradients/eval_net/l1/rnn/while/Merge_1_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switchtrain/gradients/b_count_2*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch*
_output_shapes
: : 
�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_1_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch
�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_1_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch*
_output_shapes
: 
�
9train/gradients/eval_net/l1/rnn/while/Merge_2_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*(
_output_shapes
:
:

�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_2_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_2_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
9train/gradients/eval_net/l1/rnn/while/Merge_3_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*(
_output_shapes
:
:
*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch
�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_3_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_3_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
7train/gradients/eval_net/l1/rnn/while/Enter_1_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
7train/gradients/eval_net/l1/rnn/while/Enter_2_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
7train/gradients/eval_net/l1/rnn/while/Enter_3_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
vtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntereval_net/l1/rnn/TensorArray*
is_constant(*
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations 
�
ptrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3vtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1*
sourcetrain/gradients*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes

:: 
�
ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1q^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul
�
otrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*
valueB :
���������*1
_class'
%#loc:@eval_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2otrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*1
_class'
%#loc:@eval_net/l1/rnn/while/Identity
�
ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Entereval_net/l1/rnn/while/Identity^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
qtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ktrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2qtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerl^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2o^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2w^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2u^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2w^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
`train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3ptrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ktrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:
�
_train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpN^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1a^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity`train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3`^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes

:
*
T0*s
_classi
geloc:@train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
itrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1`^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch*
_output_shapes
: 
�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*P
_classF
DBloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
dtype0*
_output_shapes
: 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
`train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulgtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
dtype0*
_output_shapes
: 
�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*
	elem_type0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*

stack_name *
_output_shapes
:
�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnter\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
btrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnter\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Mulatrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_depsNoOpY^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape[^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependencyIdentityXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshapeb^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_deps*
_output_shapes

:
*
T0*k
_classa
_]loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependency_1IdentityZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_deps*
T0*m
_classc
a_loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependency^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*
valueB :
���������*_
_classU
SQloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
dtype0*
_output_shapes
: 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*_
_classU
SQloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
`train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNeg_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
T0*
_output_shapes

:

�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependencyZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_depsNoOpY^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape[^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependencyIdentityXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshapeb^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_deps*
T0*k
_classa
_]loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
_output_shapes

:

�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependency_1IdentityZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_deps*
T0*m
_classc
a_loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_1_grad_1/NextIterationNextIterationitrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
train/gradients/AddNAddNMtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependency*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
valueB :
���������*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMultrain/gradients/AddNptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
dtype0*
_output_shapes
: 
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mulrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2train/gradients/AddN*
_output_shapes

:
*
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_deps*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:

�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
train/gradients/AddN_1AddNMtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency_1ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*
N*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumtrain/gradients/AddN_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumtrain/gradients/AddN_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*a
_classW
USloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
dtype0*
_output_shapes
: 
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
	elem_type0*a
_classW
USloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnteritrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnteritrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependencyntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulutrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapectrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_2
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_2*

stack_name *
_output_shapes
:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter eval_net/l1/rnn/while/Identity_2^train/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mulptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1Reshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_depsNoOph^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshapej^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependencyIdentitygtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshapeq^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*z
_classp
nlloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:

�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency_1Identityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*^
_classT
RPloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*^
_classT
RPloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterKeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMul|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
dtype0*
_output_shapes
: 
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mulrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGradptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_2_grad_1/NextIterationNextIterationxtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSumotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradutrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapectrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sumotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1Reshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_depsNoOph^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshapej^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependencyIdentitygtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshapeq^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_deps*
T0*z
_classp
nlloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape*
_output_shapes

:

�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependency_1Identityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGraditrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradxtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependencyqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*
N*
_output_shapes

:(*

Tidx0*
T0
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpi^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatp^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatu^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*{
_classq
omloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
~train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1Identityotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradu^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*�
_classx
vtloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnterAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMul|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
T0*
_output_shapes

:*
transpose_a( *
transpose_b(
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
	elem_type0*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*

stack_name *
_output_shapes
:
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^train/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulvtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMull^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1
�
{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMult^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_deps*
_output_shapes

:(*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1~train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:(*
T0
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationmtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes
:(*
T0
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exitptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes
:(*
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
�
ltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/Constgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeShape'eval_net/l1/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_sizeConst*
valueB :
���������*:
_class0
.,loc:@eval_net/l1/rnn/while/TensorArrayReadV3*
dtype0*
_output_shapes
: 
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_size*:
_class0
.,loc:@eval_net/l1/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/EnterEnterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter'eval_net/l1/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������*
swap_memory( 
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*'
_output_shapes
:���������*
	elem_type0
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_3
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_size*
	elem_type0*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_3*

stack_name *
_output_shapes
:
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1Enterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1 eval_net/l1/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeNShapeNttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/moditrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeNktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSlice{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN*0
_output_shapes
:������������������*
Index0*
T0
�
jtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1Slice{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:������������������*
Index0*
T0
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_depsNoOpi^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slicek^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1
�
{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slicet^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*{
_classq
omloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������
�
}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identityjtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*}
_classs
qoloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enterntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
_output_shapes

:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Mergeptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
ltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exitotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_3_grad_1/NextIterationNextIteration}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
[train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/Initializer/onesConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB(*  �?*
dtype0*
_output_shapes

:(
�
Jtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp
VariableV2*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/AssignAssignJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp[train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/readIdentityJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
^train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:(*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB(*    
�
Ltrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape
:(
�
Strain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/AssignAssignLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1^train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/readIdentityLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ytrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/Initializer/onesConst*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*  �?*
dtype0*
_output_shapes
:(
�
Htrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp
VariableV2*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/AssignAssignHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSPropYtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
�
Mtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/readIdentityHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp*
_output_shapes
:(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
�
\train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes
:(*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    
�
Jtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/AssignAssignJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1\train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros*
use_locking(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/readIdentityJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(
�
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/w2*
valueB
*  �?*
dtype0*
_output_shapes

:

�
train/eval_net/l2/w2/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l2/w2*
	container *
shape
:

�
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

�
!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
�
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/w2*
valueB
*    *
dtype0*
_output_shapes

:

�
train/eval_net/l2/w2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l2/w2*
	container *
shape
:

�
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

�
#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:*!
_class
loc:@eval_net/l2/b2*
valueB*  �?
�
train/eval_net/l2/b2/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container *
shape
:
�
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
�
!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
�
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/b2*
valueB*    *
dtype0*
_output_shapes

:
�
train/eval_net/l2/b2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container *
shape
:
�
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:*
use_locking(
�
#train/eval_net/l2/b2/RMSProp_1/readIdentitytrain/eval_net/l2/b2/RMSProp_1*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
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
�
^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyRMSPropApplyRMSProp<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSPropLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
\train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyRMSPropApplyRMSProp:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSPropJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(
�
0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
�
train/RMSPropNoOp_^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyRMSProp]^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
m
s_Placeholder*
dtype0*+
_output_shapes
:���������
* 
shape:���������

k
&target_net/l1/DropoutWrapperInit/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(target_net/l1/DropoutWrapperInit/Const_1Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(target_net/l1/DropoutWrapperInit/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:

�
^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ytarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstZtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillYtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:
*
T0
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
�
`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
[target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fill[target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:
*
T0
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

T
target_net/l1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
[
target_net/l1/range/startConst*
dtype0*
_output_shapes
: *
value	B :
[
target_net/l1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
target_net/l1/rangeRangetarget_net/l1/range/starttarget_net/l1/Ranktarget_net/l1/range/delta*

Tidx0*
_output_shapes
:
n
target_net/l1/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
[
target_net/l1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/concatConcatV2target_net/l1/concat/values_0target_net/l1/rangetarget_net/l1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
target_net/l1/transpose	Transposestarget_net/l1/concat*+
_output_shapes
:
���������*
Tperm0*
T0
n
target_net/l1/rnn/ShapeShapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
o
%target_net/l1/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
target_net/l1/rnn/strided_sliceStridedSlicetarget_net/l1/rnn/Shape%target_net/l1/rnn/strided_slice/stack'target_net/l1/rnn/strided_slice/stack_1'target_net/l1/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
target_net/l1/rnn/Shape_1Shapetarget_net/l1/transpose*
_output_shapes
:*
T0*
out_type0
q
'target_net/l1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
!target_net/l1/rnn/strided_slice_1StridedSlicetarget_net/l1/rnn/Shape_1'target_net/l1/rnn/strided_slice_1/stack)target_net/l1/rnn/strided_slice_1/stack_1)target_net/l1/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
p
target_net/l1/rnn/Shape_2Shapetarget_net/l1/transpose*
_output_shapes
:*
T0*
out_type0
q
'target_net/l1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
!target_net/l1/rnn/strided_slice_2StridedSlicetarget_net/l1/rnn/Shape_2'target_net/l1/rnn/strided_slice_2/stack)target_net/l1/rnn/strided_slice_2/stack_1)target_net/l1/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
b
 target_net/l1/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/ExpandDims
ExpandDims!target_net/l1/rnn/strided_slice_2 target_net/l1/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
a
target_net/l1/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
_
target_net/l1/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/concatConcatV2target_net/l1/rnn/ExpandDimstarget_net/l1/rnn/Consttarget_net/l1/rnn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
b
target_net/l1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/zerosFilltarget_net/l1/rnn/concattarget_net/l1/rnn/zeros/Const*
T0*'
_output_shapes
:���������

X
target_net/l1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/TensorArrayTensorArrayV3!target_net/l1/rnn/strided_slice_1*
element_shape:*
dynamic_size( *
clear_after_read(*=
tensor_array_name(&target_net/l1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
target_net/l1/rnn/TensorArray_1TensorArrayV3!target_net/l1/rnn/strided_slice_1*<
tensor_array_name'%target_net/l1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
�
*target_net/l1/rnn/TensorArrayUnstack/ShapeShapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
�
8target_net/l1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
2target_net/l1/rnn/TensorArrayUnstack/strided_sliceStridedSlice*target_net/l1/rnn/TensorArrayUnstack/Shape8target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
0target_net/l1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
r
0target_net/l1/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
*target_net/l1/rnn/TensorArrayUnstack/rangeRange0target_net/l1/rnn/TensorArrayUnstack/range/start2target_net/l1/rnn/TensorArrayUnstack/strided_slice0target_net/l1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Ltarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3target_net/l1/rnn/TensorArray_1*target_net/l1/rnn/TensorArrayUnstack/rangetarget_net/l1/transpose!target_net/l1/rnn/TensorArray_1:1*
T0**
_class 
loc:@target_net/l1/transpose*
_output_shapes
: 
�
target_net/l1/rnn/while/EnterEntertarget_net/l1/rnn/time*
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
target_net/l1/rnn/while/Enter_1Entertarget_net/l1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/Enter_2EnterXtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/Enter_3EnterZtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
_output_shapes

:
*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
target_net/l1/rnn/while/MergeMergetarget_net/l1/rnn/while/Enter%target_net/l1/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
�
target_net/l1/rnn/while/Merge_1Mergetarget_net/l1/rnn/while/Enter_1'target_net/l1/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
target_net/l1/rnn/while/Merge_2Mergetarget_net/l1/rnn/while/Enter_2'target_net/l1/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
target_net/l1/rnn/while/Merge_3Mergetarget_net/l1/rnn/while/Enter_3'target_net/l1/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
"target_net/l1/rnn/while/Less/EnterEnter!target_net/l1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/LessLesstarget_net/l1/rnn/while/Merge"target_net/l1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
b
 target_net/l1/rnn/while/LoopCondLoopCondtarget_net/l1/rnn/while/Less*
_output_shapes
: 
�
target_net/l1/rnn/while/SwitchSwitchtarget_net/l1/rnn/while/Merge target_net/l1/rnn/while/LoopCond*
_output_shapes
: : *
T0*0
_class&
$"loc:@target_net/l1/rnn/while/Merge
�
 target_net/l1/rnn/while/Switch_1Switchtarget_net/l1/rnn/while/Merge_1 target_net/l1/rnn/while/LoopCond*
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_1*
_output_shapes
: : 
�
 target_net/l1/rnn/while/Switch_2Switchtarget_net/l1/rnn/while/Merge_2 target_net/l1/rnn/while/LoopCond*
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_2*(
_output_shapes
:
:

�
 target_net/l1/rnn/while/Switch_3Switchtarget_net/l1/rnn/while/Merge_3 target_net/l1/rnn/while/LoopCond*
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_3*(
_output_shapes
:
:

o
 target_net/l1/rnn/while/IdentityIdentity target_net/l1/rnn/while/Switch:1*
T0*
_output_shapes
: 
s
"target_net/l1/rnn/while/Identity_1Identity"target_net/l1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
{
"target_net/l1/rnn/while/Identity_2Identity"target_net/l1/rnn/while/Switch_2:1*
_output_shapes

:
*
T0
{
"target_net/l1/rnn/while/Identity_3Identity"target_net/l1/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
/target_net/l1/rnn/while/TensorArrayReadV3/EnterEntertarget_net/l1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*5

frame_name'%target_net/l1/rnn/while/while_context
�
1target_net/l1/rnn/while/TensorArrayReadV3/Enter_1EnterLtarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
)target_net/l1/rnn/while/TensorArrayReadV3TensorArrayReadV3/target_net/l1/rnn/while/TensorArrayReadV3/Enter target_net/l1/rnn/while/Identity1target_net/l1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
_target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D��*
dtype0*
_output_shapes
: 
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
gtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform_target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed 
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSub]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/max]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulgtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ytarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAdd]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mul]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:(*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelYtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
Ntarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:(*O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    
�
<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name 
�
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasNtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
T0*O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
Atarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(*
T0
�
Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2)target_net/l1/rnn/while/TensorArrayReadV3"target_net/l1/rnn/while/Identity_3Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
�
Utarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnterCtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
_output_shapes

:(*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant(
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatUtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes

:(*
transpose_a( *
transpose_b( 
�
Vtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnterAtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
parallel_iterations *
_output_shapes
:(*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant(
�
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulVtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
Xtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst!^target_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitXtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst!^target_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
_output_shapes

:
*
T0
�
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoidLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0*
_output_shapes

:

�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMul"target_net/l1/rnn/while/Identity_2Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:
*
T0
�
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1SigmoidNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0*
_output_shapes

:

�
Mtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulRtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Mtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1AddLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
_output_shapes

:
*
T0
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1TanhNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes

:
*
T0
�
Ctarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst!^target_net/l1/rnn/while/Identity*
valueB"   
   *
dtype0*
_output_shapes
:
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst!^target_net/l1/rnn/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Vtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
dtype0*
_output_shapes

:
*
seed2 *

seed *
T0
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSubLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulVtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
T0*
_output_shapes

:

�
Htarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAddLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAddCtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probHtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
T0*
_output_shapes

:

�
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDivNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2Ctarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
_output_shapes

:

�
Atarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntertarget_net/l1/rnn/TensorArray*
T0*P
_classF
DBloc:@target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*5

frame_name'%target_net/l1/rnn/while/while_context
�
;target_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Atarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter target_net/l1/rnn/while/Identity=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul"target_net/l1/rnn/while/Identity_1*
_output_shapes
: *
T0*P
_classF
DBloc:@target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul
�
target_net/l1/rnn/while/add/yConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
target_net/l1/rnn/while/addAdd target_net/l1/rnn/while/Identitytarget_net/l1/rnn/while/add/y*
T0*
_output_shapes
: 
t
%target_net/l1/rnn/while/NextIterationNextIterationtarget_net/l1/rnn/while/add*
T0*
_output_shapes
: 
�
'target_net/l1/rnn/while/NextIteration_1NextIteration;target_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
'target_net/l1/rnn/while/NextIteration_2NextIterationNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
'target_net/l1/rnn/while/NextIteration_3NextIterationNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:
*
T0
e
target_net/l1/rnn/while/ExitExittarget_net/l1/rnn/while/Switch*
_output_shapes
: *
T0
i
target_net/l1/rnn/while/Exit_1Exit target_net/l1/rnn/while/Switch_1*
_output_shapes
: *
T0
q
target_net/l1/rnn/while/Exit_2Exit target_net/l1/rnn/while/Switch_2*
T0*
_output_shapes

:

q
target_net/l1/rnn/while/Exit_3Exit target_net/l1/rnn/while/Switch_3*
T0*
_output_shapes

:

�
4target_net/l1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3target_net/l1/rnn/TensorArraytarget_net/l1/rnn/while/Exit_1*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
_output_shapes
: 
�
.target_net/l1/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *0
_class&
$"loc:@target_net/l1/rnn/TensorArray
�
.target_net/l1/rnn/TensorArrayStack/range/deltaConst*
value	B :*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
(target_net/l1/rnn/TensorArrayStack/rangeRange.target_net/l1/rnn/TensorArrayStack/range/start4target_net/l1/rnn/TensorArrayStack/TensorArraySizeV3.target_net/l1/rnn/TensorArrayStack/range/delta*

Tidx0*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*#
_output_shapes
:���������
�
6target_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3target_net/l1/rnn/TensorArray(target_net/l1/rnn/TensorArrayStack/rangetarget_net/l1/rnn/while/Exit_1*
element_shape
:
*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
dtype0*"
_output_shapes
:


c
target_net/l1/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:

X
target_net/l1/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
_
target_net/l1/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
_
target_net/l1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/rangeRangetarget_net/l1/rnn/range/starttarget_net/l1/rnn/Ranktarget_net/l1/rnn/range/delta*

Tidx0*
_output_shapes
:
t
#target_net/l1/rnn/concat_1/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
a
target_net/l1/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/concat_1ConcatV2#target_net/l1/rnn/concat_1/values_0target_net/l1/rnn/rangetarget_net/l1/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
target_net/l1/rnn/transpose	Transpose6target_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3target_net/l1/rnn/concat_1*
T0*"
_output_shapes
:

*
Tperm0
v
!target_net/l1/strided_slice/stackConst*!
valueB"    ����    *
dtype0*
_output_shapes
:
x
#target_net/l1/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
x
#target_net/l1/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
target_net/l1/strided_sliceStridedSlicetarget_net/l1/rnn/transpose!target_net/l1/strided_slice/stack#target_net/l1/strided_slice/stack_1#target_net/l1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:
*
Index0*
T0
�
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
�
/target_net/l2/w2/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1target_net/l2/w2/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l2/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
?target_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l2/w2/Initializer/random_normal/shape*

seed *
T0*#
_class
loc:@target_net/l2/w2*
seed2 *
dtype0*
_output_shapes

:

�
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

�
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l2/w2
�
target_net/l2/w2
VariableV2*
shared_name *#
_class
loc:@target_net/l2/w2*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:
*
use_locking(
�
target_net/l2/w2/readIdentitytarget_net/l2/w2*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l2/w2
�
"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
target_net/l2/b2
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@target_net/l2/b2
�
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2
�
target_net/l2/b2/readIdentitytarget_net/l2/b2*
T0*#
_class
loc:@target_net/l2/b2*
_output_shapes

:
�
target_net/l2/MatMulMatMultarget_net/l1/strided_slicetarget_net/l2/w2/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
n
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*
T0*
_output_shapes

:
�
AssignAssigntarget_net/l2/w2eval_net/l2/w2/read*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:
*
use_locking(
�
Assign_1Assigntarget_net/l2/b2eval_net/l2/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2"Ҕ9�;9     >J	�`���AJ��
�.�.
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
+
Floor
x"T
y"T"
Ttype:
2
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
�
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
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
b'unknown'ۃ
l
sPlaceholder*
dtype0*+
_output_shapes
:���������
* 
shape:���������

k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
i
$eval_net/l1/DropoutWrapperInit/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
&eval_net/l1/DropoutWrapperInit/Const_1Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
&eval_net/l1/DropoutWrapperInit/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Weval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstXeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Veval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillWeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat\eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:

�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
dtype0*
_output_shapes
:*
valueB:

�
^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Yeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillYeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1^eval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
�
Xeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
R
eval_net/l1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Y
eval_net/l1/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
eval_net/l1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_net/l1/rangeRangeeval_net/l1/range/starteval_net/l1/Rankeval_net/l1/range/delta*
_output_shapes
:*

Tidx0
l
eval_net/l1/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
Y
eval_net/l1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/l1/concatConcatV2eval_net/l1/concat/values_0eval_net/l1/rangeeval_net/l1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
|
eval_net/l1/transpose	Transposeseval_net/l1/concat*
Tperm0*
T0*+
_output_shapes
:
���������
j
eval_net/l1/rnn/ShapeShapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
m
#eval_net/l1/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%eval_net/l1/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%eval_net/l1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/l1/rnn/strided_sliceStridedSliceeval_net/l1/rnn/Shape#eval_net/l1/rnn/strided_slice/stack%eval_net/l1/rnn/strided_slice/stack_1%eval_net/l1/rnn/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
l
eval_net/l1/rnn/Shape_1Shapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
o
%eval_net/l1/rnn/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
q
'eval_net/l1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/l1/rnn/strided_slice_1StridedSliceeval_net/l1/rnn/Shape_1%eval_net/l1/rnn/strided_slice_1/stack'eval_net/l1/rnn/strided_slice_1/stack_1'eval_net/l1/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
l
eval_net/l1/rnn/Shape_2Shapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
o
%eval_net/l1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'eval_net/l1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
eval_net/l1/rnn/strided_slice_2StridedSliceeval_net/l1/rnn/Shape_2%eval_net/l1/rnn/strided_slice_2/stack'eval_net/l1/rnn/strided_slice_2/stack_1'eval_net/l1/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
`
eval_net/l1/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/ExpandDims
ExpandDimseval_net/l1/rnn/strided_slice_2eval_net/l1/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
_
eval_net/l1/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
]
eval_net/l1/rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/l1/rnn/concatConcatV2eval_net/l1/rnn/ExpandDimseval_net/l1/rnn/Consteval_net/l1/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
`
eval_net/l1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/zerosFilleval_net/l1/rnn/concateval_net/l1/rnn/zeros/Const*
T0*'
_output_shapes
:���������

V
eval_net/l1/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/l1/rnn/TensorArrayTensorArrayV3eval_net/l1/rnn/strided_slice_1*
element_shape:*
dynamic_size( *
clear_after_read(*;
tensor_array_name&$eval_net/l1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
eval_net/l1/rnn/TensorArray_1TensorArrayV3eval_net/l1/rnn/strided_slice_1*
element_shape:*
dynamic_size( *
clear_after_read(*:
tensor_array_name%#eval_net/l1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
}
(eval_net/l1/rnn/TensorArrayUnstack/ShapeShapeeval_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
�
6eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
0eval_net/l1/rnn/TensorArrayUnstack/strided_sliceStridedSlice(eval_net/l1/rnn/TensorArrayUnstack/Shape6eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack8eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_18eval_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
.eval_net/l1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
p
.eval_net/l1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
(eval_net/l1/rnn/TensorArrayUnstack/rangeRange.eval_net/l1/rnn/TensorArrayUnstack/range/start0eval_net/l1/rnn/TensorArrayUnstack/strided_slice.eval_net/l1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Jeval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3eval_net/l1/rnn/TensorArray_1(eval_net/l1/rnn/TensorArrayUnstack/rangeeval_net/l1/transposeeval_net/l1/rnn/TensorArray_1:1*
T0*(
_class
loc:@eval_net/l1/transpose*
_output_shapes
: 
�
eval_net/l1/rnn/while/EnterEntereval_net/l1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/Enter_1Entereval_net/l1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/Enter_2EnterVeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
eval_net/l1/rnn/while/Enter_3EnterXeval_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/MergeMergeeval_net/l1/rnn/while/Enter#eval_net/l1/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
eval_net/l1/rnn/while/Merge_1Mergeeval_net/l1/rnn/while/Enter_1%eval_net/l1/rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0
�
eval_net/l1/rnn/while/Merge_2Mergeeval_net/l1/rnn/while/Enter_2%eval_net/l1/rnn/while/NextIteration_2*
N* 
_output_shapes
:
: *
T0
�
eval_net/l1/rnn/while/Merge_3Mergeeval_net/l1/rnn/while/Enter_3%eval_net/l1/rnn/while/NextIteration_3*
N* 
_output_shapes
:
: *
T0
�
 eval_net/l1/rnn/while/Less/EnterEntereval_net/l1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context
�
eval_net/l1/rnn/while/LessLesseval_net/l1/rnn/while/Merge eval_net/l1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
^
eval_net/l1/rnn/while/LoopCondLoopCondeval_net/l1/rnn/while/Less*
_output_shapes
: 
�
eval_net/l1/rnn/while/SwitchSwitcheval_net/l1/rnn/while/Mergeeval_net/l1/rnn/while/LoopCond*
T0*.
_class$
" loc:@eval_net/l1/rnn/while/Merge*
_output_shapes
: : 
�
eval_net/l1/rnn/while/Switch_1Switcheval_net/l1/rnn/while/Merge_1eval_net/l1/rnn/while/LoopCond*
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_1*
_output_shapes
: : 
�
eval_net/l1/rnn/while/Switch_2Switcheval_net/l1/rnn/while/Merge_2eval_net/l1/rnn/while/LoopCond*
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_2*(
_output_shapes
:
:

�
eval_net/l1/rnn/while/Switch_3Switcheval_net/l1/rnn/while/Merge_3eval_net/l1/rnn/while/LoopCond*
T0*0
_class&
$"loc:@eval_net/l1/rnn/while/Merge_3*(
_output_shapes
:
:

k
eval_net/l1/rnn/while/IdentityIdentityeval_net/l1/rnn/while/Switch:1*
T0*
_output_shapes
: 
o
 eval_net/l1/rnn/while/Identity_1Identity eval_net/l1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
w
 eval_net/l1/rnn/while/Identity_2Identity eval_net/l1/rnn/while/Switch_2:1*
T0*
_output_shapes

:

w
 eval_net/l1/rnn/while/Identity_3Identity eval_net/l1/rnn/while/Switch_3:1*
_output_shapes

:
*
T0
�
-eval_net/l1/rnn/while/TensorArrayReadV3/EnterEntereval_net/l1/rnn/TensorArray_1*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
/eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1EnterJeval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
'eval_net/l1/rnn/while/TensorArrayReadV3TensorArrayReadV3-eval_net/l1/rnn/while/TensorArrayReadV3/Entereval_net/l1/rnn/while/Identity/eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
]eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   (   
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D��*
dtype0*
_output_shapes
: 
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
eeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform]eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed 
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSub[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/max[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
�
[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMuleeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Weval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAdd[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mul[eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelWeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
�
Leval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:(*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    
�
:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
shape:(*
dtype0*
_output_shapes
:(*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container 
�
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasLeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
use_locking(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
?eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(*
T0
�
Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2'eval_net/l1/rnn/while/TensorArrayReadV3 eval_net/l1/rnn/while/Identity_3Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
Seval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnterAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
_output_shapes

:(*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatSeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
_output_shapes

:(*
transpose_a( *
transpose_b( *
T0
�
Teval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter?eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*3

frame_name%#eval_net/l1/rnn/while/while_context
�
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulTeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes

:(*
T0
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Veval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitVeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^eval_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
_output_shapes

:
*
T0
�
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoidJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0*
_output_shapes

:

�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMul eval_net/l1/rnn/while/Identity_2Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:
*
T0
�
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1SigmoidLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0*
_output_shapes

:

�
Keval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Keval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1AddJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
T0*
_output_shapes

:

�
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1TanhLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
_output_shapes

:
*
T0
�
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
Aeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probConst^eval_net/l1/rnn/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^eval_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
:*
valueB"   
   
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^eval_net/l1/rnn/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^eval_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Teval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSubJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
Jeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulTeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
T0*
_output_shapes

:

�
Feval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAddJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulJeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes

:

�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAddAeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probFeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
_output_shapes

:
*
T0
�
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
T0*
_output_shapes

:

�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDivLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2Aeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob*
T0*
_output_shapes

:

�
;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
_output_shapes

:

�
?eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntereval_net/l1/rnn/TensorArray*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
9eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3?eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entereval_net/l1/rnn/while/Identity;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul eval_net/l1/rnn/while/Identity_1*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes
: 
~
eval_net/l1/rnn/while/add/yConst^eval_net/l1/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
~
eval_net/l1/rnn/while/addAddeval_net/l1/rnn/while/Identityeval_net/l1/rnn/while/add/y*
T0*
_output_shapes
: 
p
#eval_net/l1/rnn/while/NextIterationNextIterationeval_net/l1/rnn/while/add*
T0*
_output_shapes
: 
�
%eval_net/l1/rnn/while/NextIteration_1NextIteration9eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
%eval_net/l1/rnn/while/NextIteration_2NextIterationLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
%eval_net/l1/rnn/while/NextIteration_3NextIterationLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:
*
T0
a
eval_net/l1/rnn/while/ExitExiteval_net/l1/rnn/while/Switch*
_output_shapes
: *
T0
e
eval_net/l1/rnn/while/Exit_1Exiteval_net/l1/rnn/while/Switch_1*
_output_shapes
: *
T0
m
eval_net/l1/rnn/while/Exit_2Exiteval_net/l1/rnn/while/Switch_2*
T0*
_output_shapes

:

m
eval_net/l1/rnn/while/Exit_3Exiteval_net/l1/rnn/while/Switch_3*
_output_shapes

:
*
T0
�
2eval_net/l1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3eval_net/l1/rnn/TensorArrayeval_net/l1/rnn/while/Exit_1*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
_output_shapes
: 
�
,eval_net/l1/rnn/TensorArrayStack/range/startConst*
value	B : *.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
,eval_net/l1/rnn/TensorArrayStack/range/deltaConst*
value	B :*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
&eval_net/l1/rnn/TensorArrayStack/rangeRange,eval_net/l1/rnn/TensorArrayStack/range/start2eval_net/l1/rnn/TensorArrayStack/TensorArraySizeV3,eval_net/l1/rnn/TensorArrayStack/range/delta*

Tidx0*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*#
_output_shapes
:���������
�
4eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3eval_net/l1/rnn/TensorArray&eval_net/l1/rnn/TensorArrayStack/rangeeval_net/l1/rnn/while/Exit_1*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
dtype0*"
_output_shapes
:

*
element_shape
:

a
eval_net/l1/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
V
eval_net/l1/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
]
eval_net/l1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
]
eval_net/l1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_net/l1/rnn/rangeRangeeval_net/l1/rnn/range/starteval_net/l1/rnn/Rankeval_net/l1/rnn/range/delta*

Tidx0*
_output_shapes
:
r
!eval_net/l1/rnn/concat_1/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
_
eval_net/l1/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
eval_net/l1/rnn/concat_1ConcatV2!eval_net/l1/rnn/concat_1/values_0eval_net/l1/rnn/rangeeval_net/l1/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
eval_net/l1/rnn/transpose	Transpose4eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3eval_net/l1/rnn/concat_1*
T0*"
_output_shapes
:

*
Tperm0
t
eval_net/l1/strided_slice/stackConst*!
valueB"    ����    *
dtype0*
_output_shapes
:
v
!eval_net/l1/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
v
!eval_net/l1/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
eval_net/l1/strided_sliceStridedSliceeval_net/l1/rnn/transposeeval_net/l1/strided_slice/stack!eval_net/l1/strided_slice/stack_1!eval_net/l1/strided_slice/stack_2*
end_mask*
_output_shapes

:
*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
�
.eval_net/l2/w2/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
�
-eval_net/l2/w2/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/eval_net/l2/w2/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *!
_class
loc:@eval_net/l2/w2*
valueB
 *���>
�
=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l2/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*!
_class
loc:@eval_net/l2/w2*
seed2 
�
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
�
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
�
eval_net/l2/w2
VariableV2*
shared_name *!
_class
loc:@eval_net/l2/w2*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
eval_net/l2/w2/AssignAssigneval_net/l2/w2(eval_net/l2/w2/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

{
eval_net/l2/w2/readIdentityeval_net/l2/w2*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
�
 eval_net/l2/b2/Initializer/ConstConst*!
_class
loc:@eval_net/l2/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
eval_net/l2/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container *
shape
:
�
eval_net/l2/b2/AssignAssigneval_net/l2/b2 eval_net/l2/b2/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2
{
eval_net/l2/b2/readIdentityeval_net/l2/b2*
_output_shapes

:*
T0*!
_class
loc:@eval_net/l2/b2
�
eval_net/l2/MatMulMatMuleval_net/l1/strided_sliceeval_net/l2/w2/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
h
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*
T0*
_output_shapes

:
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*'
_output_shapes
:���������*
T0
[

loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	loss/MeanMeanloss/SquaredDifference
loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
train/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
train/gradients/f_count_1Entertrain/gradients/f_count*
parallel_iterations *
_output_shapes
: *3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
train/gradients/MergeMergetrain/gradients/f_count_1train/gradients/NextIteration*
N*
_output_shapes
: : *
T0
z
train/gradients/SwitchSwitchtrain/gradients/Mergeeval_net/l1/rnn/while/LoopCond*
_output_shapes
: : *
T0
x
train/gradients/Add/yConst^eval_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
train/gradients/AddAddtrain/gradients/Switch:1train/gradients/Add/y*
T0*
_output_shapes
: 
�
train/gradients/NextIterationNextIterationtrain/gradients/Addm^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2c^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2a^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2p^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2r^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2x^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2v^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2x^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0*
_output_shapes
: 
Z
train/gradients/f_count_2Exittrain/gradients/Switch*
_output_shapes
: *
T0
Y
train/gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
�
train/gradients/b_count_1Entertrain/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
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
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
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
train/gradients/NextIteration_1NextIterationtrain/gradients/Subh^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
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
$train/gradients/loss/Mean_grad/ShapeShapeloss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
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
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/loss/SquaredDifference_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
�
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
_output_shapes

:*
T0
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
{
*train/gradients/eval_net/l2/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
�
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*
T0*
Tshape0*
_output_shapes

:
�
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
�
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape*
_output_shapes

:
�
?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
_output_shapes

:*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1
�
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
�
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/strided_slice=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
�
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul*
_output_shapes

:

�
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1*
_output_shapes

:

�
4train/gradients/eval_net/l1/strided_slice_grad/ShapeConst*!
valueB"   
   
   *
dtype0*
_output_shapes
:
�
?train/gradients/eval_net/l1/strided_slice_grad/StridedSliceGradStridedSliceGrad4train/gradients/eval_net/l1/strided_slice_grad/Shapeeval_net/l1/strided_slice/stack!eval_net/l1/strided_slice/stack_1!eval_net/l1/strided_slice/stack_2@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency*
end_mask*"
_output_shapes
:

*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
�
@train/gradients/eval_net/l1/rnn/transpose_grad/InvertPermutationInvertPermutationeval_net/l1/rnn/concat_1*
T0*
_output_shapes
:
�
8train/gradients/eval_net/l1/rnn/transpose_grad/transpose	Transpose?train/gradients/eval_net/l1/strided_slice_grad/StridedSliceGrad@train/gradients/eval_net/l1/rnn/transpose_grad/InvertPermutation*
T0*"
_output_shapes
:

*
Tperm0
�
ktrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3eval_net/l1/rnn/TensorArrayeval_net/l1/rnn/while/Exit_1*
sourcetrain/gradients*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
_output_shapes

:: 
�
gtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityeval_net/l1/rnn/while/Exit_1l^train/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*.
_class$
" loc:@eval_net/l1/rnn/TensorArray*
_output_shapes
: 
�
qtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ktrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3&eval_net/l1/rnn/TensorArrayStack/range8train/gradients/eval_net/l1/rnn/transpose_grad/transposegtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
train/gradients/zerosConst*
dtype0*
_output_shapes

:
*
valueB
*    
l
train/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
8train/gradients/eval_net/l1/rnn/while/Exit_1_grad/b_exitEnterqtrain/gradients/eval_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
8train/gradients/eval_net/l1/rnn/while/Exit_2_grad/b_exitEntertrain/gradients/zeros*
parallel_iterations *
_output_shapes

:
*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
8train/gradients/eval_net/l1/rnn/while/Exit_3_grad/b_exitEntertrain/gradients/zeros_1*
parallel_iterations *
_output_shapes

:
*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
<train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_1_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_1_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
<train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_2_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_2_grad_1/NextIteration*
N* 
_output_shapes
:
: *
T0
�
<train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switchMerge8train/gradients/eval_net/l1/rnn/while/Exit_3_grad/b_exitCtrain/gradients/eval_net/l1/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
9train/gradients/eval_net/l1/rnn/while/Merge_1_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switchtrain/gradients/b_count_2*
_output_shapes
: : *
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch
�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_1_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch*
_output_shapes
: 
�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_1_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch
�
9train/gradients/eval_net/l1/rnn/while/Merge_2_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switchtrain/gradients/b_count_2*(
_output_shapes
:
:
*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch
�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_2_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:

�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_2_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch
�
9train/gradients/eval_net/l1/rnn/while/Merge_3_grad/SwitchSwitch<train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switchtrain/gradients/b_count_2*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

�
Ctrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_depsNoOp:^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/Switch
�
Ktrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity9train/gradients/eval_net/l1/rnn/while/Merge_3_grad/SwitchD^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Mtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity;train/gradients/eval_net/l1/rnn/while/Merge_3_grad/Switch:1D^train/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch
�
7train/gradients/eval_net/l1/rnn/while/Enter_1_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
7train/gradients/eval_net/l1/rnn/while/Enter_2_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
7train/gradients/eval_net/l1/rnn/while/Enter_3_grad/ExitExitKtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
vtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntereval_net/l1/rnn/TensorArray*
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations *
is_constant(*
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3vtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1*
sourcetrain/gradients*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes

:: 
�
ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1q^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul
�
otrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*
valueB :
���������*1
_class'
%#loc:@eval_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2otrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*
	elem_type0*1
_class'
%#loc:@eval_net/l1/rnn/while/Identity*

stack_name *
_output_shapes
:
�
ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2ftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Entereval_net/l1/rnn/while/Identity^train/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
qtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterftrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ktrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2qtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^train/gradients/Sub*
_output_shapes
: *
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerl^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2o^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2w^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2u^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2w^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
`train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3ptrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ktrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2ltrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:
�
_train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpN^train/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1a^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity`train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3`^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes

:
*
T0*s
_classi
geloc:@train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
itrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityMtrain/gradients/eval_net/l1/rnn/while/Merge_1_grad/tuple/control_dependency_1`^train/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_1_grad/b_switch*
_output_shapes
: 
�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Const^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*P
_classF
DBloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
dtype0*
_output_shapes
: 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*
	elem_type0*P
_classF
DBloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*

stack_name *
_output_shapes
:
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
`train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulgtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
dtype0*
_output_shapes
: 
�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*N
_classD
B@loc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div
�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnter\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
btrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter;eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^train/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnter\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Mulatrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2gtrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_depsNoOpY^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape[^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependencyIdentityXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshapeb^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_deps*
T0*k
_classa
_]loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape*
_output_shapes

:

�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependency_1IdentityZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/group_deps*
T0*m
_classc
a_loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/ConstConst^train/gradients/Sub*
dtype0*
_output_shapes
: *
valueB
 *   ?
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependency^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*
valueB :
���������*_
_classU
SQloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
dtype0*
_output_shapes
: 
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*_
_classU
SQloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*

stack_name *
_output_shapes
:*
	elem_type0
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
`train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterLeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNeg_train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
T0*
_output_shapes

:

�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes

:

�
Ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/tuple/control_dependencyZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes

:

�
Vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumTtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeVtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1Xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
atrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_depsNoOpY^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape[^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependencyIdentityXtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshapeb^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_deps*
T0*k
_classa
_]loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
_output_shapes

:

�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependency_1IdentityZtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1b^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/group_deps*
T0*m
_classc
a_loc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1*
_output_shapes
: 
�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_1_grad_1/NextIterationNextIterationitrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
train/gradients/AddNAddNMtrain/gradients/eval_net/l1/rnn/while/Merge_3_grad/tuple/control_dependency_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/tuple/control_dependency*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
valueB :
���������*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMultrain/gradients/AddNptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
dtype0*
_output_shapes
: 
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mulrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2train/gradients/AddN*
T0*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/group_deps*
_output_shapes

:
*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
train/gradients/AddN_1AddNMtrain/gradients/eval_net/l1/rnn/while/Merge_2_grad/tuple/control_dependency_1ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*O
_classE
CAloc:@train/gradients/eval_net/l1/rnn/while/Switch_2_grad/b_switch*
N*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumtrain/gradients/AddN_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumtrain/gradients/AddN_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/group_deps*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
valueB :
���������*a
_classW
USloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
dtype0*
_output_shapes
: 
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*a
_classW
USloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnteritrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterNeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnteritrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependencyntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulutrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapectrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
valueB :
���������*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_2*
dtype0*
_output_shapes
: 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_2*

stack_name *
_output_shapes
:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter eval_net/l1/rnn/while/Identity_2^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mulptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1Reshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_depsNoOph^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshapej^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependencyIdentitygtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshapeq^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_deps*
T0*z
_classp
nlloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:

�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency_1Identityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/group_deps*
_output_shapes

:
*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^train/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shapeitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
valueB :
���������*^
_classT
RPloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
dtype0*
_output_shapes
: 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*^
_classT
RPloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterKeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMul|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
dtype0*
_output_shapes
: *
valueB :
���������*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*c
_classY
WUloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterPeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEntermtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mulrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/tuple/control_dependency_1*
_output_shapes

:
*
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Sumgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1Reshapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

�
rtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshapel^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshapes^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:

�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1s^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/group_deps*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:

�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGradptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_2_grad_1/NextIterationNextIterationxtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^train/gradients/Sub*
dtype0*
_output_shapes
:*
valueB"   
   
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^train/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shapegtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
ctrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSumotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradutrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapectrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sumetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
_output_shapes

:
*
T0*
Tshape0
�
etrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sumotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradwtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1Reshapeetrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_depsNoOph^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshapej^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependencyIdentitygtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshapeq^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_deps*
_output_shapes

:
*
T0*z
_classp
nlloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependency_1Identityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1q^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/group_deps*
_output_shapes
: *
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1
�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGraditrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradxtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/tuple/control_dependencyqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*
N*
_output_shapes

:(*

Tidx0*
T0
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes
:(*
T0
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpi^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatp^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatu^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*{
_classq
omloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
~train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1Identityotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradu^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes
:(*
T0*�
_classx
vtloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnterAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMul|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
T0*
_output_shapes

:*
transpose_a( *
transpose_b(
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
valueB :
���������*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
dtype0*
_output_shapes
: 
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*`
_classV
TRloc:@eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterMeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^train/gradients/Add*
_output_shapes

:*
swap_memory( *
T0
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^train/gradients/Sub*
_output_shapes

:*
	elem_type0
�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMulvtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpj^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMull^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1
�
{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMult^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*|
_classr
pnloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identityktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*~
_classt
rploc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:(
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Mergeqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2train/gradients/b_count_2* 
_output_shapes
:(:(*
T0
�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAddrtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1~train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationmtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exitptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes
:(*
T0
�
gtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
ltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^train/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
ftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/Constgtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeShape'eval_net/l1/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_sizeConst*
valueB :
���������*:
_class0
.,loc:@eval_net/l1/rnn/while/TensorArrayReadV3*
dtype0*
_output_shapes
: 
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2xtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc/max_size*

stack_name *
_output_shapes
:*
	elem_type0*:
_class0
.,loc:@eval_net/l1/rnn/while/TensorArrayReadV3
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/EnterEnterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter'eval_net/l1/rnn/while/TensorArrayReadV3^train/gradients/Add*
T0*'
_output_shapes
:���������*
swap_memory( 
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
ttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^train/gradients/Sub*'
_output_shapes
:���������*
	elem_type0
�
ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_sizeConst*
valueB :
���������*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_3*
dtype0*
_output_shapes
: 
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2ztrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1/max_size*

stack_name *
_output_shapes
:*
	elem_type0*3
_class)
'%loc:@eval_net/l1/rnn/while/Identity_3
�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1Enterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
parallel_iterations *
_output_shapes
:*3

frame_name%#eval_net/l1/rnn/while/while_context*
T0*
is_constant(
�
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1 eval_net/l1/rnn/while/Identity_3^train/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2|train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^train/gradients/Sub*
	elem_type0*
_output_shapes

:

�
itrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeNShapeNttrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetftrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/moditrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeNktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
htrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSlice{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetitrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:������������������
�
jtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1Slice{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependencyqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:������������������
�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_depsNoOpi^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slicek^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1
�
{train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityhtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slicet^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*{
_classq
omloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������
�
}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identityjtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1t^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*}
_classs
qoloc:@train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
ntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enterntrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*C

frame_name53train/gradients/eval_net/l1/rnn/while/while_context
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Mergeptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N* 
_output_shapes
:(: *
T0
�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2train/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
ltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAddqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
vtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationltrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
ptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exitotrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Ctrain/gradients/eval_net/l1/rnn/while/Switch_3_grad_1/NextIterationNextIteration}train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
[train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/Initializer/onesConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB(*  �?*
dtype0*
_output_shapes

:(
�
Jtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape
:(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/AssignAssignJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp[train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/Initializer/ones*
use_locking(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/readIdentityJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp*
_output_shapes

:(*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
�
^train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/Initializer/zerosConst*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
Ltrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container *
shape
:(
�
Strain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/AssignAssignLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1^train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/readIdentityLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1*
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ytrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/Initializer/onesConst*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*  �?*
dtype0*
_output_shapes
:(
�
Htrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/AssignAssignHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSPropYtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
�
Mtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/readIdentityHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(
�
\train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/Initializer/zerosConst*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
Jtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1
VariableV2*
shared_name *M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/AssignAssignJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1\train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/readIdentityJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1*
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(
�
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l2/w2*
valueB
*  �?
�
train/eval_net/l2/w2/RMSProp
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l2/w2
�
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

�
!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/w2*
valueB
*    *
dtype0*
_output_shapes

:

�
train/eval_net/l2/w2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l2/w2*
	container *
shape
:

�
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

�
#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/b2*
valueB*  �?*
dtype0*
_output_shapes

:
�
train/eval_net/l2/b2/RMSProp
VariableV2*!
_class
loc:@eval_net/l2/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2
�
!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
_output_shapes

:*
T0*!
_class
loc:@eval_net/l2/b2
�
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/b2*
valueB*    *
dtype0*
_output_shapes

:
�
train/eval_net/l2/b2/RMSProp_1
VariableV2*!
_class
loc:@eval_net/l2/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
�
#train/eval_net/l2/b2/RMSProp_1/readIdentitytrain/eval_net/l2/b2/RMSProp_1*
_output_shapes

:*
T0*!
_class
loc:@eval_net/l2/b2
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
�
^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyRMSPropApplyRMSProp<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSPropLtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonptrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*O
_classE
CAloc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
\train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyRMSPropApplyRMSProp:eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasHtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSPropJtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonqtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*M
_classC
A?loc:@eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(
�
0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

�
0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
�
train/RMSPropNoOp_^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyRMSProp]^train/RMSProp/update_eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
m
s_Placeholder* 
shape:���������
*
dtype0*+
_output_shapes
:���������

k
&target_net/l1/DropoutWrapperInit/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(target_net/l1/DropoutWrapperInit/Const_1Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(target_net/l1/DropoutWrapperInit/Const_2Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Ytarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstZtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Xtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillYtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat^target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:

�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
dtype0*
_output_shapes
:*
valueB:

�
`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
[target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fill[target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1`target_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:

�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
�
Ztarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
T
target_net/l1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
[
target_net/l1/range/startConst*
dtype0*
_output_shapes
: *
value	B :
[
target_net/l1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
target_net/l1/rangeRangetarget_net/l1/range/starttarget_net/l1/Ranktarget_net/l1/range/delta*
_output_shapes
:*

Tidx0
n
target_net/l1/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
[
target_net/l1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/concatConcatV2target_net/l1/concat/values_0target_net/l1/rangetarget_net/l1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
target_net/l1/transpose	Transposestarget_net/l1/concat*
T0*+
_output_shapes
:
���������*
Tperm0
n
target_net/l1/rnn/ShapeShapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
o
%target_net/l1/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
target_net/l1/rnn/strided_sliceStridedSlicetarget_net/l1/rnn/Shape%target_net/l1/rnn/strided_slice/stack'target_net/l1/rnn/strided_slice/stack_1'target_net/l1/rnn/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
p
target_net/l1/rnn/Shape_1Shapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
!target_net/l1/rnn/strided_slice_1StridedSlicetarget_net/l1/rnn/Shape_1'target_net/l1/rnn/strided_slice_1/stack)target_net/l1/rnn/strided_slice_1/stack_1)target_net/l1/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
p
target_net/l1/rnn/Shape_2Shapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
q
'target_net/l1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)target_net/l1/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
!target_net/l1/rnn/strided_slice_2StridedSlicetarget_net/l1/rnn/Shape_2'target_net/l1/rnn/strided_slice_2/stack)target_net/l1/rnn/strided_slice_2/stack_1)target_net/l1/rnn/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
b
 target_net/l1/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/ExpandDims
ExpandDims!target_net/l1/rnn/strided_slice_2 target_net/l1/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
a
target_net/l1/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB:

_
target_net/l1/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/concatConcatV2target_net/l1/rnn/ExpandDimstarget_net/l1/rnn/Consttarget_net/l1/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
b
target_net/l1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/zerosFilltarget_net/l1/rnn/concattarget_net/l1/rnn/zeros/Const*'
_output_shapes
:���������
*
T0
X
target_net/l1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/TensorArrayTensorArrayV3!target_net/l1/rnn/strided_slice_1*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*=
tensor_array_name(&target_net/l1/rnn/dynamic_rnn/output_0
�
target_net/l1/rnn/TensorArray_1TensorArrayV3!target_net/l1/rnn/strided_slice_1*<
tensor_array_name'%target_net/l1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
�
*target_net/l1/rnn/TensorArrayUnstack/ShapeShapetarget_net/l1/transpose*
T0*
out_type0*
_output_shapes
:
�
8target_net/l1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
2target_net/l1/rnn/TensorArrayUnstack/strided_sliceStridedSlice*target_net/l1/rnn/TensorArrayUnstack/Shape8target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_1:target_net/l1/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
0target_net/l1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
r
0target_net/l1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
*target_net/l1/rnn/TensorArrayUnstack/rangeRange0target_net/l1/rnn/TensorArrayUnstack/range/start2target_net/l1/rnn/TensorArrayUnstack/strided_slice0target_net/l1/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Ltarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3target_net/l1/rnn/TensorArray_1*target_net/l1/rnn/TensorArrayUnstack/rangetarget_net/l1/transpose!target_net/l1/rnn/TensorArray_1:1*
_output_shapes
: *
T0**
_class 
loc:@target_net/l1/transpose
�
target_net/l1/rnn/while/EnterEntertarget_net/l1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/Enter_1Entertarget_net/l1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/Enter_2EnterXtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes

:
*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant( 
�
target_net/l1/rnn/while/Enter_3EnterZtarget_net/l1/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/MergeMergetarget_net/l1/rnn/while/Enter%target_net/l1/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
target_net/l1/rnn/while/Merge_1Mergetarget_net/l1/rnn/while/Enter_1'target_net/l1/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
target_net/l1/rnn/while/Merge_2Mergetarget_net/l1/rnn/while/Enter_2'target_net/l1/rnn/while/NextIteration_2*
T0*
N* 
_output_shapes
:
: 
�
target_net/l1/rnn/while/Merge_3Mergetarget_net/l1/rnn/while/Enter_3'target_net/l1/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
"target_net/l1/rnn/while/Less/EnterEnter!target_net/l1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context
�
target_net/l1/rnn/while/LessLesstarget_net/l1/rnn/while/Merge"target_net/l1/rnn/while/Less/Enter*
_output_shapes
: *
T0
b
 target_net/l1/rnn/while/LoopCondLoopCondtarget_net/l1/rnn/while/Less*
_output_shapes
: 
�
target_net/l1/rnn/while/SwitchSwitchtarget_net/l1/rnn/while/Merge target_net/l1/rnn/while/LoopCond*
T0*0
_class&
$"loc:@target_net/l1/rnn/while/Merge*
_output_shapes
: : 
�
 target_net/l1/rnn/while/Switch_1Switchtarget_net/l1/rnn/while/Merge_1 target_net/l1/rnn/while/LoopCond*
_output_shapes
: : *
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_1
�
 target_net/l1/rnn/while/Switch_2Switchtarget_net/l1/rnn/while/Merge_2 target_net/l1/rnn/while/LoopCond*
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_2*(
_output_shapes
:
:

�
 target_net/l1/rnn/while/Switch_3Switchtarget_net/l1/rnn/while/Merge_3 target_net/l1/rnn/while/LoopCond*
T0*2
_class(
&$loc:@target_net/l1/rnn/while/Merge_3*(
_output_shapes
:
:

o
 target_net/l1/rnn/while/IdentityIdentity target_net/l1/rnn/while/Switch:1*
T0*
_output_shapes
: 
s
"target_net/l1/rnn/while/Identity_1Identity"target_net/l1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
{
"target_net/l1/rnn/while/Identity_2Identity"target_net/l1/rnn/while/Switch_2:1*
T0*
_output_shapes

:

{
"target_net/l1/rnn/while/Identity_3Identity"target_net/l1/rnn/while/Switch_3:1*
T0*
_output_shapes

:

�
/target_net/l1/rnn/while/TensorArrayReadV3/EnterEntertarget_net/l1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*5

frame_name'%target_net/l1/rnn/while/while_context
�
1target_net/l1/rnn/while/TensorArrayReadV3/Enter_1EnterLtarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant(
�
)target_net/l1/rnn/while/TensorArrayReadV3TensorArrayReadV3/target_net/l1/rnn/while/TensorArrayReadV3/Enter target_net/l1/rnn/while/Identity1target_net/l1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
_target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D��*
dtype0*
_output_shapes
: 
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�D�>
�
gtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform_target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed 
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSub]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/max]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: 
�
]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulgtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
Ytarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAdd]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mul]target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(
�
>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
	container 
�
Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelYtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*Q
_classG
ECloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
�
Ntarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes
:(*
shared_name *O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
	container *
shape:(
�
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasNtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
use_locking(*
T0*O
_classE
CAloc:@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
Atarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity<target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:(*
T0
�
Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst!^target_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2)target_net/l1/rnn/while/TensorArrayReadV3"target_net/l1/rnn/while/Identity_3Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
Utarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnterCtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
_output_shapes

:(*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*
is_constant(
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatUtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes

:(*
transpose_a( *
transpose_b( 
�
Vtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnterAtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*5

frame_name'%target_net/l1/rnn/while/while_context
�
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulVtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst!^target_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Xtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst!^target_net/l1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitXtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst!^target_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
T0*
_output_shapes

:

�
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoidLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0*
_output_shapes

:

�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMul"target_net/l1/rnn/while/Identity_2Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:
*
T0
�
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1SigmoidNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0*
_output_shapes

:

�
Mtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulRtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Mtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
_output_shapes

:
*
T0
�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1AddLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
_output_shapes

:
*
T0
�
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1TanhNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:
*
T0
�
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidPtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulOtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
Ctarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probConst!^target_net/l1/rnn/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
:*
valueB"   
   
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst!^target_net/l1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Vtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSubLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
: *
T0
�
Ltarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulVtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
T0*
_output_shapes

:

�
Htarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAddLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulLtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAddCtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_probHtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
T0*
_output_shapes

:

�
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDivNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2Ctarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob*
T0*
_output_shapes

:

�
=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
_output_shapes

:

�
Atarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntertarget_net/l1/rnn/TensorArray*
is_constant(*
_output_shapes
:*5

frame_name'%target_net/l1/rnn/while/while_context*
T0*P
_classF
DBloc:@target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations 
�
;target_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Atarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter target_net/l1/rnn/while/Identity=target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul"target_net/l1/rnn/while/Identity_1*
T0*P
_classF
DBloc:@target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes
: 
�
target_net/l1/rnn/while/add/yConst!^target_net/l1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
target_net/l1/rnn/while/addAdd target_net/l1/rnn/while/Identitytarget_net/l1/rnn/while/add/y*
T0*
_output_shapes
: 
t
%target_net/l1/rnn/while/NextIterationNextIterationtarget_net/l1/rnn/while/add*
_output_shapes
: *
T0
�
'target_net/l1/rnn/while/NextIteration_1NextIteration;target_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
'target_net/l1/rnn/while/NextIteration_2NextIterationNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:

�
'target_net/l1/rnn/while/NextIteration_3NextIterationNtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:
*
T0
e
target_net/l1/rnn/while/ExitExittarget_net/l1/rnn/while/Switch*
T0*
_output_shapes
: 
i
target_net/l1/rnn/while/Exit_1Exit target_net/l1/rnn/while/Switch_1*
T0*
_output_shapes
: 
q
target_net/l1/rnn/while/Exit_2Exit target_net/l1/rnn/while/Switch_2*
T0*
_output_shapes

:

q
target_net/l1/rnn/while/Exit_3Exit target_net/l1/rnn/while/Switch_3*
T0*
_output_shapes

:

�
4target_net/l1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3target_net/l1/rnn/TensorArraytarget_net/l1/rnn/while/Exit_1*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
_output_shapes
: 
�
.target_net/l1/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *0
_class&
$"loc:@target_net/l1/rnn/TensorArray
�
.target_net/l1/rnn/TensorArrayStack/range/deltaConst*
value	B :*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
(target_net/l1/rnn/TensorArrayStack/rangeRange.target_net/l1/rnn/TensorArrayStack/range/start4target_net/l1/rnn/TensorArrayStack/TensorArraySizeV3.target_net/l1/rnn/TensorArrayStack/range/delta*

Tidx0*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*#
_output_shapes
:���������
�
6target_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3target_net/l1/rnn/TensorArray(target_net/l1/rnn/TensorArrayStack/rangetarget_net/l1/rnn/while/Exit_1*
element_shape
:
*0
_class&
$"loc:@target_net/l1/rnn/TensorArray*
dtype0*"
_output_shapes
:


c
target_net/l1/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
X
target_net/l1/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
_
target_net/l1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
_
target_net/l1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/rangeRangetarget_net/l1/rnn/range/starttarget_net/l1/rnn/Ranktarget_net/l1/rnn/range/delta*
_output_shapes
:*

Tidx0
t
#target_net/l1/rnn/concat_1/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
a
target_net/l1/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
target_net/l1/rnn/concat_1ConcatV2#target_net/l1/rnn/concat_1/values_0target_net/l1/rnn/rangetarget_net/l1/rnn/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
target_net/l1/rnn/transpose	Transpose6target_net/l1/rnn/TensorArrayStack/TensorArrayGatherV3target_net/l1/rnn/concat_1*
T0*"
_output_shapes
:

*
Tperm0
v
!target_net/l1/strided_slice/stackConst*
dtype0*
_output_shapes
:*!
valueB"    ����    
x
#target_net/l1/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
x
#target_net/l1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
�
target_net/l1/strided_sliceStridedSlicetarget_net/l1/rnn/transpose!target_net/l1/strided_slice/stack#target_net/l1/strided_slice/stack_1#target_net/l1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes

:

�
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
�
/target_net/l2/w2/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1target_net/l2/w2/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l2/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
?target_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l2/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*#
_class
loc:@target_net/l2/w2*
seed2 
�
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l2/w2
�
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

�
target_net/l2/w2
VariableV2*#
_class
loc:@target_net/l2/w2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2
�
target_net/l2/w2/readIdentitytarget_net/l2/w2*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l2/w2
�
"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
target_net/l2/b2
VariableV2*
shared_name *#
_class
loc:@target_net/l2/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:
�
target_net/l2/b2/readIdentitytarget_net/l2/b2*
T0*#
_class
loc:@target_net/l2/b2*
_output_shapes

:
�
target_net/l2/MatMulMatMultarget_net/l1/strided_slicetarget_net/l2/w2/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
n
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*
T0*
_output_shapes

:
�
AssignAssigntarget_net/l2/w2eval_net/l2/w2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:

�
Assign_1Assigntarget_net/l2/b2eval_net/l2/b2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:""�
trainable_variables��
�
>eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignCeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Yeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Neval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
l
eval_net/l2/w2:0eval_net/l2/w2/Assigneval_net/l2/w2/read:02*eval_net/l2/w2/Initializer/random_normal:0
d
eval_net/l2/b2:0eval_net/l2/b2/Assigneval_net/l2/b2/read:02"eval_net/l2/b2/Initializer/Const:0
�
@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignEtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02[target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignCtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Ptarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
t
target_net/l2/w2:0target_net/l2/w2/Assigntarget_net/l2/w2/read:02,target_net/l2/w2/Initializer/random_normal:0
l
target_net/l2/b2:0target_net/l2/b2/Assigntarget_net/l2/b2/read:02$target_net/l2/b2/Initializer/Const:0"?
target_net_params*
(
target_net/l2/w2:0
target_net/l2/b2:0"��
while_context�ޕ
�g
#eval_net/l1/rnn/while/while_context * eval_net/l1/rnn/while/LoopCond:02eval_net/l1/rnn/while/Merge:0: eval_net/l1/rnn/while/Identity:0Beval_net/l1/rnn/while/Exit:0Beval_net/l1/rnn/while/Exit_1:0Beval_net/l1/rnn/while/Exit_2:0Beval_net/l1/rnn/while/Exit_3:0Btrain/gradients/f_count_2:0J�d
eval_net/l1/rnn/TensorArray:0
Leval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
eval_net/l1/rnn/TensorArray_1:0
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0
Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0
!eval_net/l1/rnn/strided_slice_1:0
eval_net/l1/rnn/while/Enter:0
eval_net/l1/rnn/while/Enter_1:0
eval_net/l1/rnn/while/Enter_2:0
eval_net/l1/rnn/while/Enter_3:0
eval_net/l1/rnn/while/Exit:0
eval_net/l1/rnn/while/Exit_1:0
eval_net/l1/rnn/while/Exit_2:0
eval_net/l1/rnn/while/Exit_3:0
 eval_net/l1/rnn/while/Identity:0
"eval_net/l1/rnn/while/Identity_1:0
"eval_net/l1/rnn/while/Identity_2:0
"eval_net/l1/rnn/while/Identity_3:0
"eval_net/l1/rnn/while/Less/Enter:0
eval_net/l1/rnn/while/Less:0
 eval_net/l1/rnn/while/LoopCond:0
eval_net/l1/rnn/while/Merge:0
eval_net/l1/rnn/while/Merge:1
eval_net/l1/rnn/while/Merge_1:0
eval_net/l1/rnn/while/Merge_1:1
eval_net/l1/rnn/while/Merge_2:0
eval_net/l1/rnn/while/Merge_2:1
eval_net/l1/rnn/while/Merge_3:0
eval_net/l1/rnn/while/Merge_3:1
%eval_net/l1/rnn/while/NextIteration:0
'eval_net/l1/rnn/while/NextIteration_1:0
'eval_net/l1/rnn/while/NextIteration_2:0
'eval_net/l1/rnn/while/NextIteration_3:0
eval_net/l1/rnn/while/Switch:0
eval_net/l1/rnn/while/Switch:1
 eval_net/l1/rnn/while/Switch_1:0
 eval_net/l1/rnn/while/Switch_1:1
 eval_net/l1/rnn/while/Switch_2:0
 eval_net/l1/rnn/while/Switch_2:1
 eval_net/l1/rnn/while/Switch_3:0
 eval_net/l1/rnn/while/Switch_3:1
/eval_net/l1/rnn/while/TensorArrayReadV3/Enter:0
1eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1:0
)eval_net/l1/rnn/while/TensorArrayReadV3:0
Aeval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
eval_net/l1/rnn/while/add/y:0
eval_net/l1/rnn/while/add:0
Veval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Const:0
Ueval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0
Oeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul:0
Peval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid:0
Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1:0
Reval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2:0
Meval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh:0
Oeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1:0
Teval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis:0
Oeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2:0
Xeval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dim:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:0
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2
Neval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3
?eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor:0
?eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape:0
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add:0
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div:0
Ceval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob:0
=eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul:0
Veval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul:0
Leval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub:0
Heval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform:0
train/gradients/Add/y:0
train/gradients/Add:0
train/gradients/Merge:0
train/gradients/Merge:1
train/gradients/NextIteration:0
train/gradients/Switch:0
train/gradients/Switch:1
htrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
ntrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
htrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
jtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape:0
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
wtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
ytrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc:0
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/Enter:0
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2:0
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/Enter:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc:0
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/Enter:0
utrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2:0
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/Enter:0
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2:0
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter:0
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2:0
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc:0
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter:0
btrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2:0
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc:0
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter:0
btrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2:0
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc:0
^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter:0
dtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2:0
^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc:0
train/gradients/f_count:0
train/gradients/f_count_1:0
train/gradients/f_count_2:0�
qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc:0qtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter:0�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
Leval_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:01eval_net/l1/rnn/while/TensorArrayReadV3/Enter_1:0�
Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Veval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0R
eval_net/l1/rnn/TensorArray_1:0/eval_net/l1/rnn/while/TensorArrayReadV3/Enter:0G
!eval_net/l1/rnn/strided_slice_1:0"eval_net/l1/rnn/while/Less/Enter:0�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc:0\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter:0�
ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc:0ktrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/Enter:0�
\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc:0\train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter:0�
strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0strain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc:0mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter:0�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc:0mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter:0�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/Enter:0�
^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc:0^train/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter:0�
otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0otrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/Enter:0b
eval_net/l1/rnn/TensorArray:0Aeval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Ueval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0�
mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc:0mtrain/gradients/eval_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/Enter:0�
htrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0htrain/gradients/eval_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0Reval_net/l1/rnn/while/Enter:0Reval_net/l1/rnn/while/Enter_1:0Reval_net/l1/rnn/while/Enter_2:0Reval_net/l1/rnn/while/Enter_3:0Rtrain/gradients/f_count_1:0
�.
%target_net/l1/rnn/while/while_context *"target_net/l1/rnn/while/LoopCond:02target_net/l1/rnn/while/Merge:0:"target_net/l1/rnn/while/Identity:0Btarget_net/l1/rnn/while/Exit:0B target_net/l1/rnn/while/Exit_1:0B target_net/l1/rnn/while/Exit_2:0B target_net/l1/rnn/while/Exit_3:0J�*
target_net/l1/rnn/TensorArray:0
Ntarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
!target_net/l1/rnn/TensorArray_1:0
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0
Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0
#target_net/l1/rnn/strided_slice_1:0
target_net/l1/rnn/while/Enter:0
!target_net/l1/rnn/while/Enter_1:0
!target_net/l1/rnn/while/Enter_2:0
!target_net/l1/rnn/while/Enter_3:0
target_net/l1/rnn/while/Exit:0
 target_net/l1/rnn/while/Exit_1:0
 target_net/l1/rnn/while/Exit_2:0
 target_net/l1/rnn/while/Exit_3:0
"target_net/l1/rnn/while/Identity:0
$target_net/l1/rnn/while/Identity_1:0
$target_net/l1/rnn/while/Identity_2:0
$target_net/l1/rnn/while/Identity_3:0
$target_net/l1/rnn/while/Less/Enter:0
target_net/l1/rnn/while/Less:0
"target_net/l1/rnn/while/LoopCond:0
target_net/l1/rnn/while/Merge:0
target_net/l1/rnn/while/Merge:1
!target_net/l1/rnn/while/Merge_1:0
!target_net/l1/rnn/while/Merge_1:1
!target_net/l1/rnn/while/Merge_2:0
!target_net/l1/rnn/while/Merge_2:1
!target_net/l1/rnn/while/Merge_3:0
!target_net/l1/rnn/while/Merge_3:1
'target_net/l1/rnn/while/NextIteration:0
)target_net/l1/rnn/while/NextIteration_1:0
)target_net/l1/rnn/while/NextIteration_2:0
)target_net/l1/rnn/while/NextIteration_3:0
 target_net/l1/rnn/while/Switch:0
 target_net/l1/rnn/while/Switch:1
"target_net/l1/rnn/while/Switch_1:0
"target_net/l1/rnn/while/Switch_1:1
"target_net/l1/rnn/while/Switch_2:0
"target_net/l1/rnn/while/Switch_2:1
"target_net/l1/rnn/while/Switch_3:0
"target_net/l1/rnn/while/Switch_3:1
1target_net/l1/rnn/while/TensorArrayReadV3/Enter:0
3target_net/l1/rnn/while/TensorArrayReadV3/Enter_1:0
+target_net/l1/rnn/while/TensorArrayReadV3:0
Ctarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
=target_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
target_net/l1/rnn/while/add/y:0
target_net/l1/rnn/while/add:0
Xtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Const:0
Wtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0
Qtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul:0
Rtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid:0
Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1:0
Ttarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2:0
Otarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh:0
Qtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1:0
Vtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis:0
Qtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2:0
Ztarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dim:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:0
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2
Ptarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3
Atarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor:0
Atarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape:0
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add:0
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div:0
Etarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/keep_prob:0
?target_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul:0
Xtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul:0
Ntarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub:0
Jtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform:0�
Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Wtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0�
Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Xtarget_net/l1/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0K
#target_net/l1/rnn/strided_slice_1:0$target_net/l1/rnn/while/Less/Enter:0�
Ntarget_net/l1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:03target_net/l1/rnn/while/TensorArrayReadV3/Enter_1:0f
target_net/l1/rnn/TensorArray:0Ctarget_net/l1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0V
!target_net/l1/rnn/TensorArray_1:01target_net/l1/rnn/while/TensorArrayReadV3/Enter:0Rtarget_net/l1/rnn/while/Enter:0R!target_net/l1/rnn/while/Enter_1:0R!target_net/l1/rnn/while/Enter_2:0R!target_net/l1/rnn/while/Enter_3:0"�
	variables��
�
>eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0Ceval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignCeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Yeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
<eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0Aeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAeval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Neval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
l
eval_net/l2/w2:0eval_net/l2/w2/Assigneval_net/l2/w2/read:02*eval_net/l2/w2/Initializer/random_normal:0
d
eval_net/l2/b2:0eval_net/l2/b2/Assigneval_net/l2/b2/read:02"eval_net/l2/b2/Initializer/Const:0
�
Ltrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp:0Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/AssignQtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/read:02]train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp/Initializer/ones:0
�
Ntrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1:0Strain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/AssignStrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/read:02`train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/RMSProp_1/Initializer/zeros:0
�
Jtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp:0Otrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/AssignOtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/read:02[train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp/Initializer/ones:0
�
Ltrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1:0Qtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/AssignQtrain/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/read:02^train/eval_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/RMSProp_1/Initializer/zeros:0
�
train/eval_net/l2/w2/RMSProp:0#train/eval_net/l2/w2/RMSProp/Assign#train/eval_net/l2/w2/RMSProp/read:02/train/eval_net/l2/w2/RMSProp/Initializer/ones:0
�
 train/eval_net/l2/w2/RMSProp_1:0%train/eval_net/l2/w2/RMSProp_1/Assign%train/eval_net/l2/w2/RMSProp_1/read:022train/eval_net/l2/w2/RMSProp_1/Initializer/zeros:0
�
train/eval_net/l2/b2/RMSProp:0#train/eval_net/l2/b2/RMSProp/Assign#train/eval_net/l2/b2/RMSProp/read:02/train/eval_net/l2/b2/RMSProp/Initializer/ones:0
�
 train/eval_net/l2/b2/RMSProp_1:0%train/eval_net/l2/b2/RMSProp_1/Assign%train/eval_net/l2/b2/RMSProp_1/read:022train/eval_net/l2/b2/RMSProp_1/Initializer/zeros:0
�
@target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0Etarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignEtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02[target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
>target_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0Ctarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignCtarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Ptarget_net/l1/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
t
target_net/l2/w2:0target_net/l2/w2/Assigntarget_net/l2/w2/read:02,target_net/l2/w2/Initializer/random_normal:0
l
target_net/l2/b2:0target_net/l2/b2/Assigntarget_net/l2/b2/read:02$target_net/l2/b2/Initializer/Const:0"
train_op

train/RMSProp"9
eval_net_params&
$
eval_net/l2/w2:0
eval_net/l2/b2:0p�o�