       ЃK"	  аЫжAbrain.Event:2mШЦ&1      эМK	даЫжA"Є
d
sPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Ђ
.eval_net/l1/w1/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l1/w1*
valueB"   
   *
dtype0*
_output_shapes
:

-eval_net/l1/w1/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 

/eval_net/l1/w1/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *>*
dtype0*
_output_shapes
: 
ї
=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.eval_net/l1/w1/Initializer/random_normal/shape*
T0*!
_class
loc:@eval_net/l1/w1*
seed2 *
dtype0*
_output_shapes

:
*

seed 
я
,eval_net/l1/w1/Initializer/random_normal/mulMul=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal/eval_net/l1/w1/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1
и
(eval_net/l1/w1/Initializer/random_normalAdd,eval_net/l1/w1/Initializer/random_normal/mul-eval_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1
Ѕ
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

Ю
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


 eval_net/l1/b1/Initializer/ConstConst*!
_class
loc:@eval_net/l1/b1*
valueB
*ЭЬЬ=*
dtype0*
_output_shapes

:

Ѕ
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

Ц
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


eval_net/l1/MatMulMatMulseval_net/l1/w1/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( 
q
eval_net/l1/addAddeval_net/l1/MatMuleval_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ

[
eval_net/l1/ReluRelueval_net/l1/add*'
_output_shapes
:џџџџџџџџџ
*
T0
Ђ
.eval_net/l2/w2/Initializer/random_normal/shapeConst*!
_class
loc:@eval_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:

-eval_net/l2/w2/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 

/eval_net/l2/w2/Initializer/random_normal/stddevConst*!
_class
loc:@eval_net/l2/w2*
valueB
 *>*
dtype0*
_output_shapes
: 
ї
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
я
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

и
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
Ѕ
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
Ю
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

 eval_net/l2/b2/Initializer/ConstConst*!
_class
loc:@eval_net/l2/b2*
valueB*ЭЬЬ=*
dtype0*
_output_shapes

:
Ѕ
eval_net/l2/b2
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2
Ц
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

eval_net/l2/MatMulMatMuleval_net/l1/Relueval_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
q
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*'
_output_shapes
:џџџџџџџџџ*
T0
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*'
_output_shapes
:џџџџџџџџџ*
T0
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
train/gradients/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
Ќ
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
Н
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
Љ
$train/gradients/loss/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0
ђ
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
&train/gradients/loss/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0
і
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ѕ
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
о
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
м
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 

#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
­
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
_output_shapes
:*
T0*
out_type0

3train/gradients/loss/SquaredDifference_grad/Shape_1Shapeeval_net/l2/add*
T0*
out_type0*
_output_shapes
:
џ
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
 
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ф
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Ь
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
ь
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
№
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
О
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
И
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
|
*train/gradients/eval_net/l2/add_grad/ShapeShapeeval_net/l2/MatMul*
T0*
out_type0*
_output_shapes
:
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
ъ
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ї
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ъ
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0

5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
Ђ
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
_output_shapes

:*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1
ф
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b(
к
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/Relu=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
Є
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
Ќ
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ
*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul
Љ
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1*
_output_shapes

:

Р
.train/gradients/eval_net/l1/Relu_grad/ReluGradReluGrad@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyeval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

|
*train/gradients/eval_net/l1/add_grad/ShapeShapeeval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
}
,train/gradients/eval_net/l1/add_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
ъ
:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l1/add_grad/Shape,train/gradients/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
л
(train/gradients/eval_net/l1/add_grad/SumSum.train/gradients/eval_net/l1/Relu_grad/ReluGrad:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
,train/gradients/eval_net/l1/add_grad/ReshapeReshape(train/gradients/eval_net/l1/add_grad/Sum*train/gradients/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

п
*train/gradients/eval_net/l1/add_grad/Sum_1Sum.train/gradients/eval_net/l1/Relu_grad/ReluGrad<train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ъ
.train/gradients/eval_net/l1/add_grad/Reshape_1Reshape*train/gradients/eval_net/l1/add_grad/Sum_1,train/gradients/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:


5train/gradients/eval_net/l1/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l1/add_grad/Reshape/^train/gradients/eval_net/l1/add_grad/Reshape_1
Ђ
=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l1/add_grad/Reshape6^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l1/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ


?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l1/add_grad/Reshape_16^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
_output_shapes

:
*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/add_grad/Reshape_1
ф
.train/gradients/eval_net/l1/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyeval_net/l1/w1/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ы
0train/gradients/eval_net/l1/MatMul_grad/MatMul_1MatMuls=train/gradients/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
Є
8train/gradients/eval_net/l1/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l1/MatMul_grad/MatMul1^train/gradients/eval_net/l1/MatMul_grad/MatMul_1
Ќ
@train/gradients/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l1/MatMul_grad/MatMul9^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul
Љ
Btrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l1/MatMul_grad/MatMul_19^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*C
_class9
75loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul_1
Ѕ
-train/eval_net/l1/w1/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l1/w1*
valueB
*  ?*
dtype0*
_output_shapes

:

Г
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

я
#train/eval_net/l1/w1/RMSProp/AssignAssigntrain/eval_net/l1/w1/RMSProp-train/eval_net/l1/w1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:


!train/eval_net/l1/w1/RMSProp/readIdentitytrain/eval_net/l1/w1/RMSProp*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ј
0train/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/w1*
valueB
*    
Е
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

і
%train/eval_net/l1/w1/RMSProp_1/AssignAssigntrain/eval_net/l1/w1/RMSProp_10train/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1

#train/eval_net/l1/w1/RMSProp_1/readIdentitytrain/eval_net/l1/w1/RMSProp_1*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ѕ
-train/eval_net/l1/b1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/b1*
valueB
*  ?
Г
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

я
#train/eval_net/l1/b1/RMSProp/AssignAssigntrain/eval_net/l1/b1/RMSProp-train/eval_net/l1/b1/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1

!train/eval_net/l1/b1/RMSProp/readIdentitytrain/eval_net/l1/b1/RMSProp*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/b1
Ј
0train/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l1/b1*
valueB
*    *
dtype0*
_output_shapes

:

Е
train/eval_net/l1/b1/RMSProp_1
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

і
%train/eval_net/l1/b1/RMSProp_1/AssignAssigntrain/eval_net/l1/b1/RMSProp_10train/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1

#train/eval_net/l1/b1/RMSProp_1/readIdentitytrain/eval_net/l1/b1/RMSProp_1*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

Ѕ
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l2/w2*
valueB
*  ?
Г
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
я
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:


!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
Ј
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l2/w2*
valueB
*    
Е
train/eval_net/l2/w2/RMSProp_1
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
і
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2

#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
Ѕ
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/b2*
valueB*  ?*
dtype0*
_output_shapes

:
Г
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
я
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2

!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
Ј
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*!
_class
loc:@eval_net/l2/b2*
valueB*    
Е
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
і
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:

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
з#<*
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
 *џцл.*
dtype0*
_output_shapes
: 

0train/RMSProp/update_eval_net/l1/w1/ApplyRMSPropApplyRMSPropeval_net/l1/w1train/eval_net/l1/w1/RMSProptrain/eval_net/l1/w1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:


0train/RMSProp/update_eval_net/l1/b1/ApplyRMSPropApplyRMSPropeval_net/l1/b1train/eval_net/l1/b1/RMSProptrain/eval_net/l1/b1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*!
_class
loc:@eval_net/l1/b1

0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2

0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*!
_class
loc:@eval_net/l2/b2
с
train/RMSPropNoOp1^train/RMSProp/update_eval_net/l1/w1/ApplyRMSProp1^train/RMSProp/update_eval_net/l1/b1/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
І
0target_net/l1/w1/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@target_net/l1/w1*
valueB"   
   

/target_net/l1/w1/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 

1target_net/l1/w1/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l1/w1*
valueB
 *>*
dtype0*
_output_shapes
: 
§
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
ї
.target_net/l1/w1/Initializer/random_normal/mulMul?target_net/l1/w1/Initializer/random_normal/RandomStandardNormal1target_net/l1/w1/Initializer/random_normal/stddev*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/w1
р
*target_net/l1/w1/Initializer/random_normalAdd.target_net/l1/w1/Initializer/random_normal/mul/target_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/w1
Љ
target_net/l1/w1
VariableV2*
shared_name *#
_class
loc:@target_net/l1/w1*
	container *
shape
:
*
dtype0*
_output_shapes

:

ж
target_net/l1/w1/AssignAssigntarget_net/l1/w1*target_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:


target_net/l1/w1/readIdentitytarget_net/l1/w1*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:


"target_net/l1/b1/Initializer/ConstConst*#
_class
loc:@target_net/l1/b1*
valueB
*ЭЬЬ=*
dtype0*
_output_shapes

:

Љ
target_net/l1/b1
VariableV2*
shared_name *#
_class
loc:@target_net/l1/b1*
	container *
shape
:
*
dtype0*
_output_shapes

:

Ю
target_net/l1/b1/AssignAssigntarget_net/l1/b1"target_net/l1/b1/Initializer/Const*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1*
validate_shape(*
_output_shapes

:


target_net/l1/b1/readIdentitytarget_net/l1/b1*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/b1

target_net/l1/MatMulMatMuls_target_net/l1/w1/read*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( *
T0
w
target_net/l1/addAddtarget_net/l1/MatMultarget_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ

_
target_net/l1/ReluRelutarget_net/l1/add*
T0*'
_output_shapes
:џџџџџџџџџ

І
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:

/target_net/l2/w2/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l2/w2*
valueB
 *    *
dtype0*
_output_shapes
: 

1target_net/l2/w2/Initializer/random_normal/stddevConst*#
_class
loc:@target_net/l2/w2*
valueB
 *>*
dtype0*
_output_shapes
: 
§
?target_net/l2/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l2/w2/Initializer/random_normal/shape*
T0*#
_class
loc:@target_net/l2/w2*
seed2 *
dtype0*
_output_shapes

:
*

seed 
ї
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

р
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

Љ
target_net/l2/w2
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l2/w2
ж
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:


target_net/l2/w2/readIdentitytarget_net/l2/w2*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:


"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*ЭЬЬ=*
dtype0*
_output_shapes

:
Љ
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
Ю
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2

target_net/l2/b2/readIdentitytarget_net/l2/b2*
T0*#
_class
loc:@target_net/l2/b2*
_output_shapes

:
Ё
target_net/l2/MatMulMatMultarget_net/l1/Relutarget_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
w
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
AssignAssigntarget_net/l1/w1eval_net/l1/w1/read*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1
А
Assign_1Assigntarget_net/l1/b1eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1
А
Assign_2Assigntarget_net/l2/w2eval_net/l2/w2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:

А
Assign_3Assigntarget_net/l2/b2eval_net/l2/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2"kфсЌ      Э@	ЄаЫжAJБ
чб
9
Add
x"T
y"T
z"T"
Ttype:
2	
Д
ApplyRMSProp
var"T

ms"T
mom"T
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
2	

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
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
	2	
9
Sub
x"T
y"T
z"T"
Ttype:
2	

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.4.12
b'unknown'Є
d
sPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
k
Q_targetPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
Ђ
.eval_net/l1/w1/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@eval_net/l1/w1*
valueB"   
   

-eval_net/l1/w1/Initializer/random_normal/meanConst*!
_class
loc:@eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 

/eval_net/l1/w1/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *!
_class
loc:@eval_net/l1/w1*
valueB
 *>
ї
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
я
,eval_net/l1/w1/Initializer/random_normal/mulMul=eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

и
(eval_net/l1/w1/Initializer/random_normalAdd,eval_net/l1/w1/Initializer/random_normal/mul-eval_net/l1/w1/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:

Ѕ
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

Ю
eval_net/l1/w1/AssignAssigneval_net/l1/w1(eval_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:

{
eval_net/l1/w1/readIdentityeval_net/l1/w1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1

 eval_net/l1/b1/Initializer/ConstConst*!
_class
loc:@eval_net/l1/b1*
valueB
*ЭЬЬ=*
dtype0*
_output_shapes

:

Ѕ
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

Ц
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


eval_net/l1/MatMulMatMulseval_net/l1/w1/read*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( *
T0
q
eval_net/l1/addAddeval_net/l1/MatMuleval_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ

[
eval_net/l1/ReluRelueval_net/l1/add*
T0*'
_output_shapes
:џџџџџџџџџ

Ђ
.eval_net/l2/w2/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@eval_net/l2/w2*
valueB"
      

-eval_net/l2/w2/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *!
_class
loc:@eval_net/l2/w2*
valueB
 *    

/eval_net/l2/w2/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *!
_class
loc:@eval_net/l2/w2*
valueB
 *>
ї
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
я
,eval_net/l2/w2/Initializer/random_normal/mulMul=eval_net/l2/w2/Initializer/random_normal/RandomStandardNormal/eval_net/l2/w2/Initializer/random_normal/stddev*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

и
(eval_net/l2/w2/Initializer/random_normalAdd,eval_net/l2/w2/Initializer/random_normal/mul-eval_net/l2/w2/Initializer/random_normal/mean*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

Ѕ
eval_net/l2/w2
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
Ю
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

 eval_net/l2/b2/Initializer/ConstConst*
dtype0*
_output_shapes

:*!
_class
loc:@eval_net/l2/b2*
valueB*ЭЬЬ=
Ѕ
eval_net/l2/b2
VariableV2*
shared_name *!
_class
loc:@eval_net/l2/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
Ц
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

eval_net/l2/MatMulMatMuleval_net/l1/Relueval_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
q
eval_net/l2/addAddeval_net/l2/MatMuleval_net/l2/b2/read*'
_output_shapes
:џџџџџџџџџ*
T0
x
loss/SquaredDifferenceSquaredDifferenceQ_targeteval_net/l2/add*'
_output_shapes
:џџџџџџџџџ*
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
 *  ?
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
Ќ
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/loss/Mean_grad/ShapeShapeloss/SquaredDifference*
_output_shapes
:*
T0*
out_type0
Н
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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
Љ
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
ђ
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ћ
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
і
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ѕ
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1
о
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
м
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1

#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
­
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
y
1train/gradients/loss/SquaredDifference_grad/ShapeShapeQ_target*
_output_shapes
:*
T0*
out_type0

3train/gradients/loss/SquaredDifference_grad/Shape_1Shapeeval_net/l2/add*
_output_shapes
:*
T0*
out_type0
џ
Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/loss/SquaredDifference_grad/Shape3train/gradients/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
 
2train/gradients/loss/SquaredDifference_grad/scalarConst'^train/gradients/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ф
/train/gradients/loss/SquaredDifference_grad/mulMul2train/gradients/loss/SquaredDifference_grad/scalar&train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
/train/gradients/loss/SquaredDifference_grad/subSubQ_targeteval_net/l2/add'^train/gradients/loss/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
1train/gradients/loss/SquaredDifference_grad/mul_1Mul/train/gradients/loss/SquaredDifference_grad/mul/train/gradients/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
ь
/train/gradients/loss/SquaredDifference_grad/SumSum1train/gradients/loss/SquaredDifference_grad/mul_1Atrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
3train/gradients/loss/SquaredDifference_grad/ReshapeReshape/train/gradients/loss/SquaredDifference_grad/Sum1train/gradients/loss/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
№
1train/gradients/loss/SquaredDifference_grad/Sum_1Sum1train/gradients/loss/SquaredDifference_grad/mul_1Ctrain/gradients/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
5train/gradients/loss/SquaredDifference_grad/Reshape_1Reshape1train/gradients/loss/SquaredDifference_grad/Sum_13train/gradients/loss/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

/train/gradients/loss/SquaredDifference_grad/NegNeg5train/gradients/loss/SquaredDifference_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
<train/gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp4^train/gradients/loss/SquaredDifference_grad/Reshape0^train/gradients/loss/SquaredDifference_grad/Neg
О
Dtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/loss/SquaredDifference_grad/Reshape=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*F
_class<
:8loc:@train/gradients/loss/SquaredDifference_grad/Reshape
И
Ftrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/loss/SquaredDifference_grad/Neg=^train/gradients/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*B
_class8
64loc:@train/gradients/loss/SquaredDifference_grad/Neg
|
*train/gradients/eval_net/l2/add_grad/ShapeShapeeval_net/l2/MatMul*
T0*
out_type0*
_output_shapes
:
}
,train/gradients/eval_net/l2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
ъ
:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l2/add_grad/Shape,train/gradients/eval_net/l2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ѓ
(train/gradients/eval_net/l2/add_grad/SumSumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1:train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
,train/gradients/eval_net/l2/add_grad/ReshapeReshape(train/gradients/eval_net/l2/add_grad/Sum*train/gradients/eval_net/l2/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ї
*train/gradients/eval_net/l2/add_grad/Sum_1SumFtrain/gradients/loss/SquaredDifference_grad/tuple/control_dependency_1<train/gradients/eval_net/l2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ъ
.train/gradients/eval_net/l2/add_grad/Reshape_1Reshape*train/gradients/eval_net/l2/add_grad/Sum_1,train/gradients/eval_net/l2/add_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0

5train/gradients/eval_net/l2/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l2/add_grad/Reshape/^train/gradients/eval_net/l2/add_grad/Reshape_1
Ђ
=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l2/add_grad/Reshape6^train/gradients/eval_net/l2/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@train/gradients/eval_net/l2/add_grad/Reshape

?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l2/add_grad/Reshape_16^train/gradients/eval_net/l2/add_grad/tuple/group_deps*
_output_shapes

:*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/add_grad/Reshape_1
ф
.train/gradients/eval_net/l2/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l2/add_grad/tuple/control_dependencyeval_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b(
к
0train/gradients/eval_net/l2/MatMul_grad/MatMul_1MatMuleval_net/l1/Relu=train/gradients/eval_net/l2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
Є
8train/gradients/eval_net/l2/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l2/MatMul_grad/MatMul1^train/gradients/eval_net/l2/MatMul_grad/MatMul_1
Ќ
@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l2/MatMul_grad/MatMul9^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ
*
T0*A
_class7
53loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul
Љ
Btrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l2/MatMul_grad/MatMul_19^train/gradients/eval_net/l2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/eval_net/l2/MatMul_grad/MatMul_1*
_output_shapes

:

Р
.train/gradients/eval_net/l1/Relu_grad/ReluGradReluGrad@train/gradients/eval_net/l2/MatMul_grad/tuple/control_dependencyeval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

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
ъ
:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*train/gradients/eval_net/l1/add_grad/Shape,train/gradients/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
л
(train/gradients/eval_net/l1/add_grad/SumSum.train/gradients/eval_net/l1/Relu_grad/ReluGrad:train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
,train/gradients/eval_net/l1/add_grad/ReshapeReshape(train/gradients/eval_net/l1/add_grad/Sum*train/gradients/eval_net/l1/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
п
*train/gradients/eval_net/l1/add_grad/Sum_1Sum.train/gradients/eval_net/l1/Relu_grad/ReluGrad<train/gradients/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ъ
.train/gradients/eval_net/l1/add_grad/Reshape_1Reshape*train/gradients/eval_net/l1/add_grad/Sum_1,train/gradients/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:


5train/gradients/eval_net/l1/add_grad/tuple/group_depsNoOp-^train/gradients/eval_net/l1/add_grad/Reshape/^train/gradients/eval_net/l1/add_grad/Reshape_1
Ђ
=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyIdentity,train/gradients/eval_net/l1/add_grad/Reshape6^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/eval_net/l1/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ


?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1Identity.train/gradients/eval_net/l1/add_grad/Reshape_16^train/gradients/eval_net/l1/add_grad/tuple/group_deps*
_output_shapes

:
*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/add_grad/Reshape_1
ф
.train/gradients/eval_net/l1/MatMul_grad/MatMulMatMul=train/gradients/eval_net/l1/add_grad/tuple/control_dependencyeval_net/l1/w1/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ы
0train/gradients/eval_net/l1/MatMul_grad/MatMul_1MatMuls=train/gradients/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
Є
8train/gradients/eval_net/l1/MatMul_grad/tuple/group_depsNoOp/^train/gradients/eval_net/l1/MatMul_grad/MatMul1^train/gradients/eval_net/l1/MatMul_grad/MatMul_1
Ќ
@train/gradients/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity.train/gradients/eval_net/l1/MatMul_grad/MatMul9^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul
Љ
Btrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity0train/gradients/eval_net/l1/MatMul_grad/MatMul_19^train/gradients/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*C
_class9
75loc:@train/gradients/eval_net/l1/MatMul_grad/MatMul_1
Ѕ
-train/eval_net/l1/w1/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l1/w1*
valueB
*  ?*
dtype0*
_output_shapes

:

Г
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

я
#train/eval_net/l1/w1/RMSProp/AssignAssigntrain/eval_net/l1/w1/RMSProp-train/eval_net/l1/w1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:


!train/eval_net/l1/w1/RMSProp/readIdentitytrain/eval_net/l1/w1/RMSProp*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1
Ј
0train/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/w1*
valueB
*    
Е
train/eval_net/l1/w1/RMSProp_1
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *!
_class
loc:@eval_net/l1/w1
і
%train/eval_net/l1/w1/RMSProp_1/AssignAssigntrain/eval_net/l1/w1/RMSProp_10train/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l1/w1*
validate_shape(*
_output_shapes

:


#train/eval_net/l1/w1/RMSProp_1/readIdentitytrain/eval_net/l1/w1/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/w1
Ѕ
-train/eval_net/l1/b1/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l1/b1*
valueB
*  ?*
dtype0*
_output_shapes

:

Г
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

я
#train/eval_net/l1/b1/RMSProp/AssignAssigntrain/eval_net/l1/b1/RMSProp-train/eval_net/l1/b1/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:


!train/eval_net/l1/b1/RMSProp/readIdentitytrain/eval_net/l1/b1/RMSProp*
T0*!
_class
loc:@eval_net/l1/b1*
_output_shapes

:

Ј
0train/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l1/b1*
valueB
*    
Е
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

і
%train/eval_net/l1/b1/RMSProp_1/AssignAssigntrain/eval_net/l1/b1/RMSProp_10train/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l1/b1*
validate_shape(*
_output_shapes

:


#train/eval_net/l1/b1/RMSProp_1/readIdentitytrain/eval_net/l1/b1/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l1/b1
Ѕ
-train/eval_net/l2/w2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:
*!
_class
loc:@eval_net/l2/w2*
valueB
*  ?
Г
train/eval_net/l2/w2/RMSProp
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
я
#train/eval_net/l2/w2/RMSProp/AssignAssigntrain/eval_net/l2/w2/RMSProp-train/eval_net/l2/w2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2*
validate_shape(*
_output_shapes

:


!train/eval_net/l2/w2/RMSProp/readIdentitytrain/eval_net/l2/w2/RMSProp*
T0*!
_class
loc:@eval_net/l2/w2*
_output_shapes

:

Ј
0train/eval_net/l2/w2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/w2*
valueB
*    *
dtype0*
_output_shapes

:

Е
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
і
%train/eval_net/l2/w2/RMSProp_1/AssignAssigntrain/eval_net/l2/w2/RMSProp_10train/eval_net/l2/w2/RMSProp_1/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@eval_net/l2/w2

#train/eval_net/l2/w2/RMSProp_1/readIdentitytrain/eval_net/l2/w2/RMSProp_1*
_output_shapes

:
*
T0*!
_class
loc:@eval_net/l2/w2
Ѕ
-train/eval_net/l2/b2/RMSProp/Initializer/onesConst*!
_class
loc:@eval_net/l2/b2*
valueB*  ?*
dtype0*
_output_shapes

:
Г
train/eval_net/l2/b2/RMSProp
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *!
_class
loc:@eval_net/l2/b2
я
#train/eval_net/l2/b2/RMSProp/AssignAssigntrain/eval_net/l2/b2/RMSProp-train/eval_net/l2/b2/RMSProp/Initializer/ones*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:

!train/eval_net/l2/b2/RMSProp/readIdentitytrain/eval_net/l2/b2/RMSProp*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
Ј
0train/eval_net/l2/b2/RMSProp_1/Initializer/zerosConst*!
_class
loc:@eval_net/l2/b2*
valueB*    *
dtype0*
_output_shapes

:
Е
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
і
%train/eval_net/l2/b2/RMSProp_1/AssignAssigntrain/eval_net/l2/b2/RMSProp_10train/eval_net/l2/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@eval_net/l2/b2*
validate_shape(*
_output_shapes

:

#train/eval_net/l2/b2/RMSProp_1/readIdentitytrain/eval_net/l2/b2/RMSProp_1*
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
`
train/RMSProp/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
X
train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
[
train/RMSProp/momentumConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
train/RMSProp/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *џцл.

0train/RMSProp/update_eval_net/l1/w1/ApplyRMSPropApplyRMSPropeval_net/l1/w1train/eval_net/l1/w1/RMSProptrain/eval_net/l1/w1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l1/w1*
_output_shapes

:


0train/RMSProp/update_eval_net/l1/b1/ApplyRMSPropApplyRMSPropeval_net/l1/b1train/eval_net/l1/b1/RMSProptrain/eval_net/l1/b1/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l1/add_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*!
_class
loc:@eval_net/l1/b1

0train/RMSProp/update_eval_net/l2/w2/ApplyRMSPropApplyRMSPropeval_net/l2/w2train/eval_net/l2/w2/RMSProptrain/eval_net/l2/w2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilonBtrain/gradients/eval_net/l2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*!
_class
loc:@eval_net/l2/w2

0train/RMSProp/update_eval_net/l2/b2/ApplyRMSPropApplyRMSPropeval_net/l2/b2train/eval_net/l2/b2/RMSProptrain/eval_net/l2/b2/RMSProp_1train/RMSProp/learning_ratetrain/RMSProp/decaytrain/RMSProp/momentumtrain/RMSProp/epsilon?train/gradients/eval_net/l2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@eval_net/l2/b2*
_output_shapes

:
с
train/RMSPropNoOp1^train/RMSProp/update_eval_net/l1/w1/ApplyRMSProp1^train/RMSProp/update_eval_net/l1/b1/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/w2/ApplyRMSProp1^train/RMSProp/update_eval_net/l2/b2/ApplyRMSProp
e
s_Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
І
0target_net/l1/w1/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l1/w1*
valueB"   
   *
dtype0*
_output_shapes
:

/target_net/l1/w1/Initializer/random_normal/meanConst*#
_class
loc:@target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 

1target_net/l1/w1/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *#
_class
loc:@target_net/l1/w1*
valueB
 *>
§
?target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal0target_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:
*

seed *
T0*#
_class
loc:@target_net/l1/w1*
seed2 
ї
.target_net/l1/w1/Initializer/random_normal/mulMul?target_net/l1/w1/Initializer/random_normal/RandomStandardNormal1target_net/l1/w1/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:

р
*target_net/l1/w1/Initializer/random_normalAdd.target_net/l1/w1/Initializer/random_normal/mul/target_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/w1
Љ
target_net/l1/w1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l1/w1*
	container *
shape
:

ж
target_net/l1/w1/AssignAssigntarget_net/l1/w1*target_net/l1/w1/Initializer/random_normal*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1

target_net/l1/w1/readIdentitytarget_net/l1/w1*
T0*#
_class
loc:@target_net/l1/w1*
_output_shapes

:


"target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:
*#
_class
loc:@target_net/l1/b1*
valueB
*ЭЬЬ=
Љ
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

Ю
target_net/l1/b1/AssignAssigntarget_net/l1/b1"target_net/l1/b1/Initializer/Const*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1*
validate_shape(*
_output_shapes

:


target_net/l1/b1/readIdentitytarget_net/l1/b1*
_output_shapes

:
*
T0*#
_class
loc:@target_net/l1/b1

target_net/l1/MatMulMatMuls_target_net/l1/w1/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( 
w
target_net/l1/addAddtarget_net/l1/MatMultarget_net/l1/b1/read*'
_output_shapes
:џџџџџџџџџ
*
T0
_
target_net/l1/ReluRelutarget_net/l1/add*'
_output_shapes
:џџџџџџџџџ
*
T0
І
0target_net/l2/w2/Initializer/random_normal/shapeConst*#
_class
loc:@target_net/l2/w2*
valueB"
      *
dtype0*
_output_shapes
:

/target_net/l2/w2/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *#
_class
loc:@target_net/l2/w2*
valueB
 *    

1target_net/l2/w2/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *#
_class
loc:@target_net/l2/w2*
valueB
 *>
§
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
ї
.target_net/l2/w2/Initializer/random_normal/mulMul?target_net/l2/w2/Initializer/random_normal/RandomStandardNormal1target_net/l2/w2/Initializer/random_normal/stddev*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

р
*target_net/l2/w2/Initializer/random_normalAdd.target_net/l2/w2/Initializer/random_normal/mul/target_net/l2/w2/Initializer/random_normal/mean*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:

Љ
target_net/l2/w2
VariableV2*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *#
_class
loc:@target_net/l2/w2
ж
target_net/l2/w2/AssignAssigntarget_net/l2/w2*target_net/l2/w2/Initializer/random_normal*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:


target_net/l2/w2/readIdentitytarget_net/l2/w2*
T0*#
_class
loc:@target_net/l2/w2*
_output_shapes

:


"target_net/l2/b2/Initializer/ConstConst*#
_class
loc:@target_net/l2/b2*
valueB*ЭЬЬ=*
dtype0*
_output_shapes

:
Љ
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
Ю
target_net/l2/b2/AssignAssigntarget_net/l2/b2"target_net/l2/b2/Initializer/Const*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2*
validate_shape(*
_output_shapes

:

target_net/l2/b2/readIdentitytarget_net/l2/b2*
_output_shapes

:*
T0*#
_class
loc:@target_net/l2/b2
Ё
target_net/l2/MatMulMatMultarget_net/l1/Relutarget_net/l2/w2/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
w
target_net/l2/addAddtarget_net/l2/MatMultarget_net/l2/b2/read*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
AssignAssigntarget_net/l1/w1eval_net/l1/w1/read*
use_locking(*
T0*#
_class
loc:@target_net/l1/w1*
validate_shape(*
_output_shapes

:

А
Assign_1Assigntarget_net/l1/b1eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*#
_class
loc:@target_net/l1/b1
А
Assign_2Assigntarget_net/l2/w2eval_net/l2/w2/read*
use_locking(*
T0*#
_class
loc:@target_net/l2/w2*
validate_shape(*
_output_shapes

:

А
Assign_3Assigntarget_net/l2/b2eval_net/l2/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@target_net/l2/b2""
trainable_variablesѓ№
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
eval_net/l2/b2:0"
	variables
l
eval_net/l1/w1:0eval_net/l1/w1/Assigneval_net/l1/w1/read:02*eval_net/l1/w1/Initializer/random_normal:0
d
eval_net/l1/b1:0eval_net/l1/b1/Assigneval_net/l1/b1/read:02"eval_net/l1/b1/Initializer/Const:0
l
eval_net/l2/w2:0eval_net/l2/w2/Assigneval_net/l2/w2/read:02*eval_net/l2/w2/Initializer/random_normal:0
d
eval_net/l2/b2:0eval_net/l2/b2/Assigneval_net/l2/b2/read:02"eval_net/l2/b2/Initializer/Const:0

train/eval_net/l1/w1/RMSProp:0#train/eval_net/l1/w1/RMSProp/Assign#train/eval_net/l1/w1/RMSProp/read:02/train/eval_net/l1/w1/RMSProp/Initializer/ones:0
Є
 train/eval_net/l1/w1/RMSProp_1:0%train/eval_net/l1/w1/RMSProp_1/Assign%train/eval_net/l1/w1/RMSProp_1/read:022train/eval_net/l1/w1/RMSProp_1/Initializer/zeros:0

train/eval_net/l1/b1/RMSProp:0#train/eval_net/l1/b1/RMSProp/Assign#train/eval_net/l1/b1/RMSProp/read:02/train/eval_net/l1/b1/RMSProp/Initializer/ones:0
Є
 train/eval_net/l1/b1/RMSProp_1:0%train/eval_net/l1/b1/RMSProp_1/Assign%train/eval_net/l1/b1/RMSProp_1/read:022train/eval_net/l1/b1/RMSProp_1/Initializer/zeros:0

train/eval_net/l2/w2/RMSProp:0#train/eval_net/l2/w2/RMSProp/Assign#train/eval_net/l2/w2/RMSProp/read:02/train/eval_net/l2/w2/RMSProp/Initializer/ones:0
Є
 train/eval_net/l2/w2/RMSProp_1:0%train/eval_net/l2/w2/RMSProp_1/Assign%train/eval_net/l2/w2/RMSProp_1/read:022train/eval_net/l2/w2/RMSProp_1/Initializer/zeros:0

train/eval_net/l2/b2/RMSProp:0#train/eval_net/l2/b2/RMSProp/Assign#train/eval_net/l2/b2/RMSProp/read:02/train/eval_net/l2/b2/RMSProp/Initializer/ones:0
Є
 train/eval_net/l2/b2/RMSProp_1:0%train/eval_net/l2/b2/RMSProp_1/Assign%train/eval_net/l2/b2/RMSProp_1/read:022train/eval_net/l2/b2/RMSProp_1/Initializer/zeros:0
t
target_net/l1/w1:0target_net/l1/w1/Assigntarget_net/l1/w1/read:02,target_net/l1/w1/Initializer/random_normal:0
l
target_net/l1/b1:0target_net/l1/b1/Assigntarget_net/l1/b1/read:02$target_net/l1/b1/Initializer/Const:0
t
target_net/l2/w2:0target_net/l2/w2/Assigntarget_net/l2/w2/read:02,target_net/l2/w2/Initializer/random_normal:0
l
target_net/l2/b2:0target_net/l2/b2/Assigntarget_net/l2/b2/read:02$target_net/l2/b2/Initializer/Const:0ссp