       £K"	  АЎЋ÷Abrain.Event:2JНуі1В      нЉKМ	gQ∞ЎЋ÷A"§Д
d
sPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
Ґ
.eval_net/l1/w1/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l1/w1*
valueB"   
   *
dtype0*
_output_shapes
:
Х
-eval_net/l1/w1/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
/eval_net/l1/w1/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
ч
=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*!
_class
loc:@eval_net/l1/w1*
seed2 
п
,eval_net/l1/w1/Initializer/random_normal/mulMul=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ў
(eval_net/l1/w1/Initializer/random_normalAdd,eval_net/l1/w1/Initializer/random_normal/mul-eval_net/l1/w1/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

•
eval_net/l1/w1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:

ќ
eval_net/l1/w1/AssignAssigneval_net/l1/w1(eval_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:

{
eval_net/l1/w1/readIdentityeval_net/l1/w1*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ш
 eval_net/l1/b1/Initializer/ConstConst*!
_class
loc:@eval_net/l1/b1*
valueB
*Ќћћ=*
dtype0*
_output_shapes

:

•
eval_net/l1/b1
VariableV2*
shared_name *!
_class
loc:@eval_net/l1/b1*
	container *
shape
:
*
dtype0*
_output_shapes

:

∆
eval_net/l1/b1/AssignAssigneval_net/l1/b1 eval_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1
{
eval_net/l1/b1/readIdentityeval_net/l1/b1*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

М
eval_net/l1/MatMulMatMulseval_net/l1/w1/read*
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b( 
q
eval_net/l1/addAddeval_net/l1/MatMuleval_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€
*
T0
[
eval_net/l1/ReluRelueval_net/l1/add*
T0*'
_output_shapes
:€€€€€€€€€

Ґ
.eval_net/l2/w2/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
Х
-eval_net/l2/w2/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
/eval_net/l2/w2/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
ч
=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l2/w2/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:
*

seed *
T0*!
_class
loc:@eval_net/l2/w2
п
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

Ў
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

•
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
ќ
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
Ш
 eval_net/l2/b2/Initializer/ConstConst*
dtype0*
_output_shapes

:*!
_class
loc:@eval_net/l2/b2*
valueB*Ќћћ=
•
eval_net/l2/b2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container 
∆
eval_net/l2/b2/AssignAssigneval_net/l2/b2 eval_net/l2/b2/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2
{
eval_net/l2/b2/readIdentityeval_net/l2/b2*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
Ы
eval_net/l2/MatMulMatMuleval_net/l1/Relueval_net/l2/w2/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
q
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*
T0*'
_output_shapes
:€€€€€€€€€
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*'
_output_shapes
:€€€€€€€€€*
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
train/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
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
љ
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
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
©
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
т
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ђ
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
ц
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
•
(train/gradients/loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0
ё
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
№
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *
T0
Д
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
≠
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
T0*
out_type0*
_output_shapes
:
В
3train/gradients/loss/SquaredDifference_grad/Shape_1Shapeeval_net/l2/add*
T0*
out_type0*
_output_shapes
:
€
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
†
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
ƒ
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ђ
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
ћ
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
м
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
в
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
р
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
и
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Я
/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
ђ
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
Њ
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Є
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg
|
*train/gradients/eval_net/l2/add_grad/ShapeShapeeval_net/l2/MatMul*
_output_shapes
:*
T0*
out_type0
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
к
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
у
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ќ
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ч
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
 
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Э
5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
Ґ
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Я
?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1*
_output_shapes

:
д
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b(*
T0
Џ
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/Relu=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
§
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
ђ
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€

©
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1*
_output_shapes

:

ј
.train/gradients/eval_net/l1/Relu_grad/ReluGradReluGrad@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyeval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€

|
*train/gradients/eval_net/l1/add_grad/ShapeShapeeval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
}
,train/gradients/eval_net/l1/add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
к
:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l1/add_grad/Shape,train/gradients/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
џ
(train/gradients/eval_net/l1/add_grad/SumSum.train/gradients/eval_net/l1/Relu_grad/ReluGrad:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ќ
,train/gradients/eval_net/l1/add_grad/ReshapeReshape(train/gradients/eval_net/l1/add_grad/Sum*train/gradients/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

я
*train/gradients/eval_net/l1/add_grad/Sum_1Sum.train/gradients/eval_net/l1/Relu_grad/ReluGrad<train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
 
.train/gradients/eval_net/l1/add_grad/Reshape_1Reshape*train/gradients/eval_net/l1/add_grad/Sum_1,train/gradients/eval_net/l1/add_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
Э
5train/gradients/eval_net/l1/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l1/add_grad/Reshape/^train/gradients/eval_net/l1/add_grad/Reshape_1
Ґ
=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l1/add_grad/Reshape6^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l1/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€

Я
?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l1/add_grad/Reshape_16^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:

д
.train/gradients/eval_net/l1/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyeval_net/l1/w1/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(*
T0
Ћ
0train/gradients/eval_net/l1/MatMul_grad/MatMul_1MatMuls=train/gradients/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
§
8train/gradients/eval_net/l1/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l1/MatMul_grad/MatMul1^train/gradients/eval_net/l1/MatMul_grad/MatMul_1
ђ
@train/gradients/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l1/MatMul_grad/MatMul9^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
©
Btrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l1/MatMul_grad/MatMul_19^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:

•
-train/eval_net/l1/w1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/w1*
valueB
*  А?
≥
train/eval_net/l1/w1/RMSProp
VariableV2*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:
*
dtype0*
_output_shapes

:

п
#train/eval_net/l1/w1/RMSProp/AssignAssigntrain/eval_net/l1/w1/RMSProp-train/eval_net/l1/w1/RMSProp/Initializer/ones*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:
*
use_locking(
Ч
!train/eval_net/l1/w1/RMSProp/readIdentitytrain/eval_net/l1/w1/RMSProp*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

®
0train/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l1/w1*
valueB
*    *
dtype0*
_output_shapes

:

µ
train/eval_net/l1/w1/RMSProp_1
VariableV2*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:
*
dtype0*
_output_shapes

:

ц
%train/eval_net/l1/w1/RMSProp_1/AssignAssigntrain/eval_net/l1/w1/RMSProp_10train/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1
Ы
#train/eval_net/l1/w1/RMSProp_1/readIdentitytrain/eval_net/l1/w1/RMSProp_1*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

•
-train/eval_net/l1/b1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/b1*
valueB
*  А?
≥
train/eval_net/l1/b1/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/b1*
	container *
shape
:

п
#train/eval_net/l1/b1/RMSProp/AssignAssigntrain/eval_net/l1/b1/RMSProp-train/eval_net/l1/b1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:

Ч
!train/eval_net/l1/b1/RMSProp/readIdentitytrain/eval_net/l1/b1/RMSProp*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

®
0train/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l1/b1*
valueB
*    *
dtype0*
_output_shapes

:

µ
train/eval_net/l1/b1/RMSProp_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/b1*
	container 
ц
%train/eval_net/l1/b1/RMSProp_1/AssignAssigntrain/eval_net/l1/b1/RMSProp_10train/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:
*
use_locking(
Ы
#train/eval_net/l1/b1/RMSProp_1/readIdentitytrain/eval_net/l1/b1/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/b1
•
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/w2*
valueB
*  А?*
dtype0*
_output_shapes

:

≥
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
п
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2
Ч
!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

®
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/w2*
valueB
*    *
dtype0*
_output_shapes

:

µ
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
ц
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

Ы
#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

•
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:*!
_class
loc:@eval_net/l2/b2*
valueB*  А?
≥
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
п
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
Ч
!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
_output_shapes

:*
T0*!
_class
loc:@eval_net/l2/b2
®
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/b2*
valueB*    *
dtype0*
_output_shapes

:
µ
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
ц
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
Ы
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
„#<*
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
 *€жџ.*
dtype0*
_output_shapes
: 
Т
0train/RMSProp/update_eval_net/l1/w1/ApplyRMSPropApplyRMSPropeval_net/l1/w1train/eval_net/l1/w1/RMSProptrain/eval_net/l1/w1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

П
0train/RMSProp/update_eval_net/l1/b1/ApplyRMSPropApplyRMSPropeval_net/l1/b1train/eval_net/l1/b1/RMSProptrain/eval_net/l1/b1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

Т
0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2
П
0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
б
train/RMSPropNoOp1^train/RMSProp/update_eval_net/l1/w1/ApplyRMSProp1^train/RMSProp/update_eval_net/l1/b1/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
¶
0target_net/l1/w1/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@target_net/l1/w1*
valueB"   
   
Щ
/target_net/l1/w1/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
1target_net/l1/w1/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l1/w1*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
э
?target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l1/w1/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:
*

seed *
T0*#
_class
loc:@target_net/l1/w1
ч
.target_net/l1/w1/Initializer/random_normal/mulMul?target_net/l1/w1/Initializer/random_normal/RandomStandardNormal1target_net/l1/w1/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:

а
*target_net/l1/w1/Initializer/random_normalAdd.target_net/l1/w1/Initializer/random_normal/mul/target_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/w1
©
target_net/l1/w1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l1/w1*
	container 
÷
target_net/l1/w1/AssignAssigntarget_net/l1/w1*target_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:

Б
target_net/l1/w1/readIdentitytarget_net/l1/w1*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:

Ь
"target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:
*#
_class
loc:@target_net/l1/b1*
valueB
*Ќћћ=
©
target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l1/b1*
	container *
shape
:

ќ
target_net/l1/b1/AssignAssigntarget_net/l1/b1"target_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1
Б
target_net/l1/b1/readIdentitytarget_net/l1/b1*
T0*#
_class
loc:@target_net/l1/b1*
_output_shapes

:

С
target_net/l1/MatMulMatMuls_target_net/l1/w1/read*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b( *
T0
w
target_net/l1/addAddtarget_net/l1/MatMultarget_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€
*
T0
_
target_net/l1/ReluRelutarget_net/l1/add*
T0*'
_output_shapes
:€€€€€€€€€

¶
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
Щ
/target_net/l2/w2/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *#
_class
loc:@target_net/l2/w2*
valueB
 *    
Ы
1target_net/l2/w2/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l2/w2*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
э
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
ч
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

а
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

©
target_net/l2/w2
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l2/w2*
	container *
shape
:

÷
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:

Б
target_net/l2/w2/readIdentitytarget_net/l2/w2*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

Ь
"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*Ќћћ=*
dtype0*
_output_shapes

:
©
target_net/l2/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@target_net/l2/b2*
	container *
shape
:
ќ
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:*
use_locking(
Б
target_net/l2/b2/readIdentitytarget_net/l2/b2*
_output_shapes

:*
T0*#
_class
loc:@target_net/l2/b2
°
target_net/l2/MatMulMatMultarget_net/l1/Relutarget_net/l2/w2/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
w
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*'
_output_shapes
:€€€€€€€€€*
T0
Ѓ
AssignAssigntarget_net/l1/w1eval_net/l1/w1/read*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:
*
use_locking(
∞
Assign_1Assigntarget_net/l1/b1eval_net/l1/b1/read*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1*
validate_shape(*
_output_shapes

:

∞
Assign_2Assigntarget_net/l2/w2eval_net/l2/w2/read*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:
*
use_locking(
∞
Assign_3Assigntarget_net/l2/b2eval_net/l2/b2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:"-•ЏђШ      Ќ@	а∞ЎЋ÷AJЯ±
з—
9
Add
x"T
y"T
z"T"
Ttype:
2	
і
ApplyRMSProp
var"TА

ms"TА
mom"TА
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
.
Identity

input"T
output"T"	
Ttype
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
2	Р
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
Д
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
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
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	Р
9
Sub
x"T
y"T
z"T"
Ttype:
2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.4.12
b'unknown'§Д
d
sPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
Ґ
.eval_net/l1/w1/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l1/w1*
valueB"   
   *
dtype0*
_output_shapes
:
Х
-eval_net/l1/w1/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
/eval_net/l1/w1/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
ч
=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*!
_class
loc:@eval_net/l1/w1*
seed2 
п
,eval_net/l1/w1/Initializer/random_normal/mulMul=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ў
(eval_net/l1/w1/Initializer/random_normalAdd,eval_net/l1/w1/Initializer/random_normal/mul-eval_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1
•
eval_net/l1/w1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:

ќ
eval_net/l1/w1/AssignAssigneval_net/l1/w1(eval_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:

{
eval_net/l1/w1/readIdentityeval_net/l1/w1*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ш
 eval_net/l1/b1/Initializer/ConstConst*!
_class
loc:@eval_net/l1/b1*
valueB
*Ќћћ=*
dtype0*
_output_shapes

:

•
eval_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/b1*
	container *
shape
:

∆
eval_net/l1/b1/AssignAssigneval_net/l1/b1 eval_net/l1/b1/Initializer/Const*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:

{
eval_net/l1/b1/readIdentityeval_net/l1/b1*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

М
eval_net/l1/MatMulMatMulseval_net/l1/w1/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
q
eval_net/l1/addAddeval_net/l1/MatMuleval_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€
*
T0
[
eval_net/l1/ReluRelueval_net/l1/add*
T0*'
_output_shapes
:€€€€€€€€€

Ґ
.eval_net/l2/w2/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
Х
-eval_net/l2/w2/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
/eval_net/l2/w2/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
ч
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
п
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
Ў
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

•
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
ќ
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
Ш
 eval_net/l2/b2/Initializer/ConstConst*!
_class
loc:@eval_net/l2/b2*
valueB*Ќћћ=*
dtype0*
_output_shapes

:
•
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
∆
eval_net/l2/b2/AssignAssigneval_net/l2/b2 eval_net/l2/b2/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2
{
eval_net/l2/b2/readIdentityeval_net/l2/b2*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
Ы
eval_net/l2/MatMulMatMuleval_net/l1/Relueval_net/l2/w2/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
q
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*
T0*'
_output_shapes
:€€€€€€€€€
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*
T0*'
_output_shapes
:€€€€€€€€€
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
 *  А?
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
}
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
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
љ
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
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
©
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
т
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ђ
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
ц
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
•
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
ё
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
№
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
Д
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
≠
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
T0*
out_type0*
_output_shapes
:
В
3train/gradients/loss/SquaredDifference_grad/Shape_1Shapeeval_net/l2/add*
T0*
out_type0*
_output_shapes
:
€
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
†
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
ƒ
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ђ
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
ћ
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
м
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
в
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
р
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
и
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Я
/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
ђ
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
Њ
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape
Є
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
|
*train/gradients/eval_net/l2/add_grad/ShapeShapeeval_net/l2/MatMul*
_output_shapes
:*
T0*
out_type0
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
к
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
у
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ќ
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ч
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
 
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Э
5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
Ґ
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Я
?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
_output_shapes

:*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1
д
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
Џ
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/Relu=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
§
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
ђ
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€

©
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1
ј
.train/gradients/eval_net/l1/Relu_grad/ReluGradReluGrad@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyeval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€
*
T0
|
*train/gradients/eval_net/l1/add_grad/ShapeShapeeval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
}
,train/gradients/eval_net/l1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
к
:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l1/add_grad/Shape,train/gradients/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
џ
(train/gradients/eval_net/l1/add_grad/SumSum.train/gradients/eval_net/l1/Relu_grad/ReluGrad:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ќ
,train/gradients/eval_net/l1/add_grad/ReshapeReshape(train/gradients/eval_net/l1/add_grad/Sum*train/gradients/eval_net/l1/add_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
T0*
Tshape0
я
*train/gradients/eval_net/l1/add_grad/Sum_1Sum.train/gradients/eval_net/l1/Relu_grad/ReluGrad<train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
 
.train/gradients/eval_net/l1/add_grad/Reshape_1Reshape*train/gradients/eval_net/l1/add_grad/Sum_1,train/gradients/eval_net/l1/add_grad/Shape_1*
_output_shapes

:
*
T0*
Tshape0
Э
5train/gradients/eval_net/l1/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l1/add_grad/Reshape/^train/gradients/eval_net/l1/add_grad/Reshape_1
Ґ
=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l1/add_grad/Reshape6^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l1/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€

Я
?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l1/add_grad/Reshape_16^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
_output_shapes

:
*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/add_grad/Reshape_1
д
.train/gradients/eval_net/l1/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyeval_net/l1/w1/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
Ћ
0train/gradients/eval_net/l1/MatMul_grad/MatMul_1MatMuls=train/gradients/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
§
8train/gradients/eval_net/l1/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l1/MatMul_grad/MatMul1^train/gradients/eval_net/l1/MatMul_grad/MatMul_1
ђ
@train/gradients/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l1/MatMul_grad/MatMul9^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul
©
Btrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l1/MatMul_grad/MatMul_19^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*C
_class9
75loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul_1
•
-train/eval_net/l1/w1/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l1/w1*
valueB
*  А?*
dtype0*
_output_shapes

:

≥
train/eval_net/l1/w1/RMSProp
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:

п
#train/eval_net/l1/w1/RMSProp/AssignAssigntrain/eval_net/l1/w1/RMSProp-train/eval_net/l1/w1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:

Ч
!train/eval_net/l1/w1/RMSProp/readIdentitytrain/eval_net/l1/w1/RMSProp*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

®
0train/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/w1*
valueB
*    
µ
train/eval_net/l1/w1/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/w1*
	container *
shape
:

ц
%train/eval_net/l1/w1/RMSProp_1/AssignAssigntrain/eval_net/l1/w1/RMSProp_10train/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1
Ы
#train/eval_net/l1/w1/RMSProp_1/readIdentitytrain/eval_net/l1/w1/RMSProp_1*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

•
-train/eval_net/l1/b1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/b1*
valueB
*  А?
≥
train/eval_net/l1/b1/RMSProp
VariableV2*!
_class
loc:@eval_net/l1/b1*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
п
#train/eval_net/l1/b1/RMSProp/AssignAssigntrain/eval_net/l1/b1/RMSProp-train/eval_net/l1/b1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:

Ч
!train/eval_net/l1/b1/RMSProp/readIdentitytrain/eval_net/l1/b1/RMSProp*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

®
0train/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l1/b1*
valueB
*    *
dtype0*
_output_shapes

:

µ
train/eval_net/l1/b1/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/b1*
	container *
shape
:

ц
%train/eval_net/l1/b1/RMSProp_1/AssignAssigntrain/eval_net/l1/b1/RMSProp_10train/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1
Ы
#train/eval_net/l1/b1/RMSProp_1/readIdentitytrain/eval_net/l1/b1/RMSProp_1*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

•
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/w2*
valueB
*  А?*
dtype0*
_output_shapes

:

≥
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
п
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:
*
use_locking(
Ч
!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

®
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/w2*
valueB
*    *
dtype0*
_output_shapes

:

µ
train/eval_net/l2/w2/RMSProp_1
VariableV2*!
_class
loc:@eval_net/l2/w2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
ц
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:

Ы
#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
•
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/b2*
valueB*  А?*
dtype0*
_output_shapes

:
≥
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
п
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
Ч
!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
®
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/b2*
valueB*    *
dtype0*
_output_shapes

:
µ
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
ц
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:
Ы
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
„#<*
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
train/RMSProp/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *€жџ.
Т
0train/RMSProp/update_eval_net/l1/w1/ApplyRMSPropApplyRMSPropeval_net/l1/w1train/eval_net/l1/w1/RMSProptrain/eval_net/l1/w1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

П
0train/RMSProp/update_eval_net/l1/b1/ApplyRMSPropApplyRMSPropeval_net/l1/b1train/eval_net/l1/b1/RMSProptrain/eval_net/l1/b1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

Т
0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

П
0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:*
use_locking( 
б
train/RMSPropNoOp1^train/RMSProp/update_eval_net/l1/w1/ApplyRMSProp1^train/RMSProp/update_eval_net/l1/b1/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
¶
0target_net/l1/w1/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l1/w1*
valueB"   
   *
dtype0*
_output_shapes
:
Щ
/target_net/l1/w1/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
1target_net/l1/w1/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l1/w1*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
э
?target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l1/w1/Initializer/random_normal/shape*
T0*#
_class
loc:@target_net/l1/w1*
seed2 *
dtype0*
_output_shapes

:
*

seed 
ч
.target_net/l1/w1/Initializer/random_normal/mulMul?target_net/l1/w1/Initializer/random_normal/RandomStandardNormal1target_net/l1/w1/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:

а
*target_net/l1/w1/Initializer/random_normalAdd.target_net/l1/w1/Initializer/random_normal/mul/target_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/w1
©
target_net/l1/w1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l1/w1*
	container 
÷
target_net/l1/w1/AssignAssigntarget_net/l1/w1*target_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:

Б
target_net/l1/w1/readIdentitytarget_net/l1/w1*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:

Ь
"target_net/l1/b1/Initializer/ConstConst*#
_class
loc:@target_net/l1/b1*
valueB
*Ќћћ=*
dtype0*
_output_shapes

:

©
target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l1/b1*
	container *
shape
:

ќ
target_net/l1/b1/AssignAssigntarget_net/l1/b1"target_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1
Б
target_net/l1/b1/readIdentitytarget_net/l1/b1*
T0*#
_class
loc:@target_net/l1/b1*
_output_shapes

:

С
target_net/l1/MatMulMatMuls_target_net/l1/w1/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
w
target_net/l1/addAddtarget_net/l1/MatMultarget_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€

_
target_net/l1/ReluRelutarget_net/l1/add*
T0*'
_output_shapes
:€€€€€€€€€

¶
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:
Щ
/target_net/l2/w2/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
1target_net/l2/w2/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l2/w2*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: 
э
?target_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l2/w2/Initializer/random_normal/shape*
seed2 *
dtype0*
_output_shapes

:
*

seed *
T0*#
_class
loc:@target_net/l2/w2
ч
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

а
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l2/w2
©
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
÷
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:
*
use_locking(
Б
target_net/l2/w2/readIdentitytarget_net/l2/w2*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

Ь
"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*Ќћћ=*
dtype0*
_output_shapes

:
©
target_net/l2/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@target_net/l2/b2*
	container *
shape
:
ќ
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:
Б
target_net/l2/b2/readIdentitytarget_net/l2/b2*
_output_shapes

:*
T0*#
_class
loc:@target_net/l2/b2
°
target_net/l2/MatMulMatMultarget_net/l1/Relutarget_net/l2/w2/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
w
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*
T0*'
_output_shapes
:€€€€€€€€€
Ѓ
AssignAssigntarget_net/l1/w1eval_net/l1/w1/read*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:
*
use_locking(
∞
Assign_1Assigntarget_net/l1/b1eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1
∞
Assign_2Assigntarget_net/l2/w2eval_net/l2/w2/read*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2
∞
Assign_3Assigntarget_net/l2/b2eval_net/l2/b2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:""Л
trainable_variablesур
l
eval_net/l1/w1:0eval_net/l1/w1/Assigneval_net/l1/w1/read:02*eval_net/l1/w1/Initializer/random_normal:0
d
eval_net/l1/b1:0eval_net/l1/b1/Assigneval_net/l1/b1/read:02"eval_net/l1/b1/Initializer/Const:0
l
eval_net/l2/w2:0eval_net/l2/w2/Assigneval_net/l2/w2/read:02*eval_net/l2/w2/Initializer/random_normal:0
d
eval_net/l2/b2:0eval_net/l2/b2/Assigneval_net/l2/b2/read:02"eval_net/l2/b2/Initializer/Const:0
t
target_net/l1/w1:0target_net/l1/w1/Assigntarget_net/l1/w1/read:02,target_net/l1/w1/Initializer/random_normal:0
l
target_net/l1/b1:0target_net/l1/b1/Assigntarget_net/l1/b1/read:02$target_net/l1/b1/Initializer/Const:0
t
target_net/l2/w2:0target_net/l2/w2/Assigntarget_net/l2/w2/read:02,target_net/l2/w2/Initializer/random_normal:0
l
target_net/l2/b2:0target_net/l2/b2/Assigntarget_net/l2/b2/read:02$target_net/l2/b2/Initializer/Const:0"g
target_net_paramsR
P
target_net/l1/w1:0
target_net/l1/b1:0
target_net/l2/w2:0
target_net/l2/b2:0"
train_op

train/RMSProp"]
eval_net_paramsJ
H
eval_net/l1/w1:0
eval_net/l1/b1:0
eval_net/l2/w2:0
eval_net/l2/b2:0"Х
	variablesЗД
l
eval_net/l1/w1:0eval_net/l1/w1/Assigneval_net/l1/w1/read:02*eval_net/l1/w1/Initializer/random_normal:0
d
eval_net/l1/b1:0eval_net/l1/b1/Assigneval_net/l1/b1/read:02"eval_net/l1/b1/Initializer/Const:0
l
eval_net/l2/w2:0eval_net/l2/w2/Assigneval_net/l2/w2/read:02*eval_net/l2/w2/Initializer/random_normal:0
d
eval_net/l2/b2:0eval_net/l2/b2/Assigneval_net/l2/b2/read:02"eval_net/l2/b2/Initializer/Const:0
Ы
train/eval_net/l1/w1/RMSProp:0#train/eval_net/l1/w1/RMSProp/Assign#train/eval_net/l1/w1/RMSProp/read:02/train/eval_net/l1/w1/RMSProp/Initializer/ones:0
§
 train/eval_net/l1/w1/RMSProp_1:0%train/eval_net/l1/w1/RMSProp_1/Assign%train/eval_net/l1/w1/RMSProp_1/read:022train/eval_net/l1/w1/RMSProp_1/Initializer/zeros:0
Ы
train/eval_net/l1/b1/RMSProp:0#train/eval_net/l1/b1/RMSProp/Assign#train/eval_net/l1/b1/RMSProp/read:02/train/eval_net/l1/b1/RMSProp/Initializer/ones:0
§
 train/eval_net/l1/b1/RMSProp_1:0%train/eval_net/l1/b1/RMSProp_1/Assign%train/eval_net/l1/b1/RMSProp_1/read:022train/eval_net/l1/b1/RMSProp_1/Initializer/zeros:0
Ы
train/eval_net/l2/w2/RMSProp:0#train/eval_net/l2/w2/RMSProp/Assign#train/eval_net/l2/w2/RMSProp/read:02/train/eval_net/l2/w2/RMSProp/Initializer/ones:0
§
 train/eval_net/l2/w2/RMSProp_1:0%train/eval_net/l2/w2/RMSProp_1/Assign%train/eval_net/l2/w2/RMSProp_1/read:022train/eval_net/l2/w2/RMSProp_1/Initializer/zeros:0
Ы
train/eval_net/l2/b2/RMSProp:0#train/eval_net/l2/b2/RMSProp/Assign#train/eval_net/l2/b2/RMSProp/read:02/train/eval_net/l2/b2/RMSProp/Initializer/ones:0
§
 train/eval_net/l2/b2/RMSProp_1:0%train/eval_net/l2/b2/RMSProp_1/Assign%train/eval_net/l2/b2/RMSProp_1/read:022train/eval_net/l2/b2/RMSProp_1/Initializer/zeros:0
t
target_net/l1/w1:0target_net/l1/w1/Assigntarget_net/l1/w1/read:02,target_net/l1/w1/Initializer/random_normal:0
l
target_net/l1/b1:0target_net/l1/b1/Assigntarget_net/l1/b1/read:02$target_net/l1/b1/Initializer/Const:0
t
target_net/l2/w2:0target_net/l2/w2/Assigntarget_net/l2/w2/read:02,target_net/l2/w2/Initializer/random_normal:0
l
target_net/l2/b2:0target_net/l2/b2/Assigntarget_net/l2/b2/read:02$target_net/l2/b2/Initializer/Const:0Р»9S