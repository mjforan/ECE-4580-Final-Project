öØ-
Ģ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878%

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:@*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_10/kernel
~
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*'
_output_shapes
:@*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:*
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:*
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:*
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel

$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:*
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel

$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*(
_output_shapes
:*
dtype0
u
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
n
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes	
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_8/kernel/m

*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_9/kernel/m

*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/m

6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/m

5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_10/kernel/m

+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/m
|
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_8/gamma/m

6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_8/beta/m

5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/m

+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/m
|
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/m

6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/m

5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/m

+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/m
|
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_10/gamma/m

7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_10/beta/m

6Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_13/kernel/m

+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_13/bias/m
|
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/m

7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/m

6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_14/kernel/m

+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_14/bias/m
|
)Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/m

7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/m

6Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_15/kernel/m

+Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_15/bias/m
|
)Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_13/gamma/m

7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_13/beta/m

6Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_8/kernel/v

*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_9/kernel/v

*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/v

6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/v

5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_10/kernel/v

+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/v
|
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_8/gamma/v

6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_8/beta/v

5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/v

+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/v
|
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_9/gamma/v

6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_9/beta/v

5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/v

+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/v
|
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_10/gamma/v

7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_10/beta/v

6Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_13/kernel/v

+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_13/bias/v
|
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/v

7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/v

6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_14/kernel/v

+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_14/bias/v
|
)Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/v

7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/v

6Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_15/kernel/v

+Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_15/bias/v
|
)Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_13/gamma/v

7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_13/beta/v

6Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Āź
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*üé
valueńéBķé Båé

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
layer-29
layer_with_weights-18
layer-30
 	optimizer
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
h

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api

2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
R
;trainable_variables
<	variables
=regularization_losses
>	keras_api
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api

Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
h

Rkernel
Sbias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api

Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]trainable_variables
^	variables
_regularization_losses
`	keras_api
R
atrainable_variables
b	variables
cregularization_losses
d	keras_api
R
etrainable_variables
f	variables
gregularization_losses
h	keras_api
h

ikernel
jbias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api

oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
h

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api

~axis
	gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
 trainable_variables
”	variables
¢regularization_losses
£	keras_api
 
	¤axis

„gamma
	¦beta
§moving_mean
Ømoving_variance
©trainable_variables
Ŗ	variables
«regularization_losses
¬	keras_api
V
­trainable_variables
®	variables
Æregularization_losses
°	keras_api
V
±trainable_variables
²	variables
³regularization_losses
“	keras_api
V
µtrainable_variables
¶	variables
·regularization_losses
ø	keras_api
n
¹kernel
	ŗbias
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api
V
ætrainable_variables
Ą	variables
Įregularization_losses
Ā	keras_api
n
Ćkernel
	Äbias
Åtrainable_variables
Ę	variables
Ēregularization_losses
Č	keras_api
V
Étrainable_variables
Ź	variables
Ėregularization_losses
Ģ	keras_api
n
Ķkernel
	Ībias
Ļtrainable_variables
Š	variables
Ńregularization_losses
Ņ	keras_api
V
Ótrainable_variables
Ō	variables
Õregularization_losses
Ö	keras_api
n
×kernel
	Ųbias
Łtrainable_variables
Ś	variables
Ūregularization_losses
Ü	keras_api
ß
	Żiter
Žbeta_1
ßbeta_2

ądecay
įlearning_rate&m'm,m-m3m4mCmDmJmKmRmSmYmZmimjmpmqmxmym m”	m¢	m£	m¤	m„	m¦	m§	mØ	„m©	¦mŖ	¹m«	ŗm¬	Ćm­	Äm®	ĶmÆ	Īm°	×m±	Ųm²&v³'v“,vµ-v¶3v·4vøCv¹DvŗJv»Kv¼Rv½Sv¾YvæZvĄivĮjvĀpvĆqvÄxvÅyvĘvĒ	vČ	vÉ	vŹ	vĖ	vĢ	vĶ	vĪ	„vĻ	¦vŠ	¹vŃ	ŗvŅ	ĆvÓ	ÄvŌ	ĶvÕ	ĪvÖ	×v×	ŲvŲ
·
&0
'1
,2
-3
34
45
C6
D7
J8
K9
R10
S11
Y12
Z13
i14
j15
p16
q17
x18
y19
20
21
22
23
24
25
26
27
„28
¦29
¹30
ŗ31
Ć32
Ä33
Ķ34
Ī35
×36
Ų37
­
&0
'1
,2
-3
34
45
56
67
C8
D9
J10
K11
L12
M13
R14
S15
Y16
Z17
[18
\19
i20
j21
p22
q23
r24
s25
x26
y27
28
29
30
31
32
33
34
35
36
37
38
39
„40
¦41
§42
Ø43
¹44
ŗ45
Ć46
Ä47
Ķ48
Ī49
×50
Ų51
 
²
ānon_trainable_variables
ćlayer_metrics
!trainable_variables
"	variables
ämetrics
#regularization_losses
 ålayer_regularization_losses
ęlayers
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
²
ēnon_trainable_variables
člayer_metrics
(trainable_variables
)	variables
émetrics
*regularization_losses
 źlayer_regularization_losses
ėlayers
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
²
ģnon_trainable_variables
ķlayer_metrics
.trainable_variables
/	variables
īmetrics
0regularization_losses
 ļlayer_regularization_losses
šlayers
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
52
63
 
²
ńnon_trainable_variables
ņlayer_metrics
7trainable_variables
8	variables
ómetrics
9regularization_losses
 ōlayer_regularization_losses
õlayers
 
 
 
²
önon_trainable_variables
÷layer_metrics
;trainable_variables
<	variables
ųmetrics
=regularization_losses
 łlayer_regularization_losses
ślayers
 
 
 
²
ūnon_trainable_variables
ülayer_metrics
?trainable_variables
@	variables
żmetrics
Aregularization_losses
 žlayer_regularization_losses
’layers
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
²
non_trainable_variables
layer_metrics
Etrainable_variables
F	variables
metrics
Gregularization_losses
 layer_regularization_losses
layers
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
L2
M3
 
²
non_trainable_variables
layer_metrics
Ntrainable_variables
O	variables
metrics
Pregularization_losses
 layer_regularization_losses
layers
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
 
²
non_trainable_variables
layer_metrics
Ttrainable_variables
U	variables
metrics
Vregularization_losses
 layer_regularization_losses
layers
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
[2
\3
 
²
non_trainable_variables
layer_metrics
]trainable_variables
^	variables
metrics
_regularization_losses
 layer_regularization_losses
layers
 
 
 
²
non_trainable_variables
layer_metrics
atrainable_variables
b	variables
metrics
cregularization_losses
 layer_regularization_losses
layers
 
 
 
²
non_trainable_variables
layer_metrics
etrainable_variables
f	variables
metrics
gregularization_losses
 layer_regularization_losses
layers
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

i0
j1
 
²
non_trainable_variables
layer_metrics
ktrainable_variables
l	variables
 metrics
mregularization_losses
 ”layer_regularization_losses
¢layers
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1

p0
q1
r2
s3
 
²
£non_trainable_variables
¤layer_metrics
ttrainable_variables
u	variables
„metrics
vregularization_losses
 ¦layer_regularization_losses
§layers
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
 
²
Ønon_trainable_variables
©layer_metrics
ztrainable_variables
{	variables
Ŗmetrics
|regularization_losses
 «layer_regularization_losses
¬layers
 
hf
VARIABLE_VALUEbatch_normalization_11/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
3
 
µ
­non_trainable_variables
®layer_metrics
trainable_variables
	variables
Æmetrics
regularization_losses
 °layer_regularization_losses
±layers
 
 
 
µ
²non_trainable_variables
³layer_metrics
trainable_variables
	variables
“metrics
regularization_losses
 µlayer_regularization_losses
¶layers
 
 
 
µ
·non_trainable_variables
ølayer_metrics
trainable_variables
	variables
¹metrics
regularization_losses
 ŗlayer_regularization_losses
»layers
][
VARIABLE_VALUEconv2d_14/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_14/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
¼non_trainable_variables
½layer_metrics
trainable_variables
	variables
¾metrics
regularization_losses
 ælayer_regularization_losses
Ąlayers
 
hf
VARIABLE_VALUEbatch_normalization_12/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_12/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_12/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_12/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
0
1
2
3
 
µ
Įnon_trainable_variables
Ālayer_metrics
trainable_variables
	variables
Ćmetrics
regularization_losses
 Älayer_regularization_losses
Ålayers
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Ęnon_trainable_variables
Ēlayer_metrics
 trainable_variables
”	variables
Čmetrics
¢regularization_losses
 Élayer_regularization_losses
Źlayers
 
hf
VARIABLE_VALUEbatch_normalization_13/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_13/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_13/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_13/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

„0
¦1
 
„0
¦1
§2
Ø3
 
µ
Ėnon_trainable_variables
Ģlayer_metrics
©trainable_variables
Ŗ	variables
Ķmetrics
«regularization_losses
 Īlayer_regularization_losses
Ļlayers
 
 
 
µ
Šnon_trainable_variables
Ńlayer_metrics
­trainable_variables
®	variables
Ņmetrics
Æregularization_losses
 Ólayer_regularization_losses
Ōlayers
 
 
 
µ
Õnon_trainable_variables
Ölayer_metrics
±trainable_variables
²	variables
×metrics
³regularization_losses
 Ųlayer_regularization_losses
Łlayers
 
 
 
µ
Śnon_trainable_variables
Ūlayer_metrics
µtrainable_variables
¶	variables
Ümetrics
·regularization_losses
 Żlayer_regularization_losses
Žlayers
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

¹0
ŗ1

¹0
ŗ1
 
µ
ßnon_trainable_variables
ąlayer_metrics
»trainable_variables
¼	variables
įmetrics
½regularization_losses
 ālayer_regularization_losses
ćlayers
 
 
 
µ
änon_trainable_variables
ålayer_metrics
ætrainable_variables
Ą	variables
ęmetrics
Įregularization_losses
 ēlayer_regularization_losses
člayers
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

Ć0
Ä1

Ć0
Ä1
 
µ
énon_trainable_variables
źlayer_metrics
Åtrainable_variables
Ę	variables
ėmetrics
Ēregularization_losses
 ģlayer_regularization_losses
ķlayers
 
 
 
µ
īnon_trainable_variables
ļlayer_metrics
Étrainable_variables
Ź	variables
šmetrics
Ėregularization_losses
 ńlayer_regularization_losses
ņlayers
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

Ķ0
Ī1

Ķ0
Ī1
 
µ
ónon_trainable_variables
ōlayer_metrics
Ļtrainable_variables
Š	variables
õmetrics
Ńregularization_losses
 ölayer_regularization_losses
÷layers
 
 
 
µ
ųnon_trainable_variables
łlayer_metrics
Ótrainable_variables
Ō	variables
śmetrics
Õregularization_losses
 ūlayer_regularization_losses
ülayers
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

×0
Ų1

×0
Ų1
 
µ
żnon_trainable_variables
žlayer_metrics
Łtrainable_variables
Ś	variables
’metrics
Ūregularization_losses
 layer_regularization_losses
layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
l
50
61
L2
M3
[4
\5
r6
s7
8
9
10
11
§12
Ø13
 

0
1
 
ī
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 
 
 
 
 
 

[0
\1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

§0
Ø1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_8/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_9/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_12/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_10/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_10/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_11/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_14/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_14/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_12/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_15/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_15/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_13/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_4/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_4/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_5/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_5/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_6/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_6/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_7/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_7/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_8/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_9/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_12/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_10/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_10/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_11/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_14/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_14/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_12/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_12/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_15/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_15/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_13/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_13/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_4/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_4/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_5/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_5/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_6/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_6/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_7/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_7/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_8_inputPlaceholder*/
_output_shapes
:’’’’’’’’’00*
dtype0*$
shape:’’’’’’’’’00
Ķ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_8_inputconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_238815
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
·4
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_10/beta/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp)Adam/conv2d_14/bias/m/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_12/beta/m/Read/ReadVariableOp+Adam/conv2d_15/kernel/m/Read/ReadVariableOp)Adam/conv2d_15/bias/m/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_13/beta/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_10/beta/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp)Adam/conv2d_14/bias/v/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_12/beta/v/Read/ReadVariableOp+Adam/conv2d_15/kernel/v/Read/ReadVariableOp)Adam/conv2d_15/bias/v/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_13/beta/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_241299
ž
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/m#Adam/batch_normalization_10/gamma/m"Adam/batch_normalization_10/beta/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/conv2d_14/kernel/mAdam/conv2d_14/bias/m#Adam/batch_normalization_12/gamma/m"Adam/batch_normalization_12/beta/mAdam/conv2d_15/kernel/mAdam/conv2d_15/bias/m#Adam/batch_normalization_13/gamma/m"Adam/batch_normalization_13/beta/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/v#Adam/batch_normalization_10/gamma/v"Adam/batch_normalization_10/beta/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/vAdam/conv2d_14/kernel/vAdam/conv2d_14/bias/v#Adam/batch_normalization_12/gamma/v"Adam/batch_normalization_12/beta/vAdam/conv2d_15/kernel/vAdam/conv2d_15/bias/v#Adam/batch_normalization_13/gamma/v"Adam/batch_normalization_13/beta/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_241720 
Ż

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_237650

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240242

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_4_layer_call_fn_236296

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_2362902
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_237875

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_5_layer_call_fn_236516

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2365102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
’

-__inference_sequential_1_layer_call_fn_239397

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*H
_read_only_resource_inputs*
(&	
!"#$'()*-./01234*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2383292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs

G
+__inference_dropout_11_layer_call_fn_240740

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
®
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239642

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
„

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_236389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_237401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_237111

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
	
­
E__inference_conv2d_10_layer_call_and_return_conditional_losses_239724

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’@:::W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_236578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
„

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
±
«
C__inference_dense_4_layer_call_and_return_conditional_losses_240704

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_237573

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_237823

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

”
-__inference_sequential_1_layer_call_fn_238436
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*H
_read_only_resource_inputs*
(&	
!"#$'()*-./01234*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2383292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input
¦
©
6__inference_batch_normalization_8_layer_call_fn_239861

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2363892
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
©
d
+__inference_dropout_11_layer_call_fn_240735

inputs
identity¢StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_9_layer_call_and_return_conditional_losses_239549

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’..@:::W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
·
F
*__inference_dropout_7_layer_call_fn_239713

inputs
identityĪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ļ
¬
D__inference_conv2d_8_layer_call_and_return_conditional_losses_236977

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
ReluĶ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’00:::W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
É
ņM
"__inference__traced_restore_241720
file_prefix$
 assignvariableop_conv2d_8_kernel$
 assignvariableop_1_conv2d_8_bias&
"assignvariableop_2_conv2d_9_kernel$
 assignvariableop_3_conv2d_9_bias2
.assignvariableop_4_batch_normalization_7_gamma1
-assignvariableop_5_batch_normalization_7_beta8
4assignvariableop_6_batch_normalization_7_moving_mean<
8assignvariableop_7_batch_normalization_7_moving_variance'
#assignvariableop_8_conv2d_10_kernel%
!assignvariableop_9_conv2d_10_bias3
/assignvariableop_10_batch_normalization_8_gamma2
.assignvariableop_11_batch_normalization_8_beta9
5assignvariableop_12_batch_normalization_8_moving_mean=
9assignvariableop_13_batch_normalization_8_moving_variance(
$assignvariableop_14_conv2d_11_kernel&
"assignvariableop_15_conv2d_11_bias3
/assignvariableop_16_batch_normalization_9_gamma2
.assignvariableop_17_batch_normalization_9_beta9
5assignvariableop_18_batch_normalization_9_moving_mean=
9assignvariableop_19_batch_normalization_9_moving_variance(
$assignvariableop_20_conv2d_12_kernel&
"assignvariableop_21_conv2d_12_bias4
0assignvariableop_22_batch_normalization_10_gamma3
/assignvariableop_23_batch_normalization_10_beta:
6assignvariableop_24_batch_normalization_10_moving_mean>
:assignvariableop_25_batch_normalization_10_moving_variance(
$assignvariableop_26_conv2d_13_kernel&
"assignvariableop_27_conv2d_13_bias4
0assignvariableop_28_batch_normalization_11_gamma3
/assignvariableop_29_batch_normalization_11_beta:
6assignvariableop_30_batch_normalization_11_moving_mean>
:assignvariableop_31_batch_normalization_11_moving_variance(
$assignvariableop_32_conv2d_14_kernel&
"assignvariableop_33_conv2d_14_bias4
0assignvariableop_34_batch_normalization_12_gamma3
/assignvariableop_35_batch_normalization_12_beta:
6assignvariableop_36_batch_normalization_12_moving_mean>
:assignvariableop_37_batch_normalization_12_moving_variance(
$assignvariableop_38_conv2d_15_kernel&
"assignvariableop_39_conv2d_15_bias4
0assignvariableop_40_batch_normalization_13_gamma3
/assignvariableop_41_batch_normalization_13_beta:
6assignvariableop_42_batch_normalization_13_moving_mean>
:assignvariableop_43_batch_normalization_13_moving_variance&
"assignvariableop_44_dense_4_kernel$
 assignvariableop_45_dense_4_bias&
"assignvariableop_46_dense_5_kernel$
 assignvariableop_47_dense_5_bias&
"assignvariableop_48_dense_6_kernel$
 assignvariableop_49_dense_6_bias&
"assignvariableop_50_dense_7_kernel$
 assignvariableop_51_dense_7_bias!
assignvariableop_52_adam_iter#
assignvariableop_53_adam_beta_1#
assignvariableop_54_adam_beta_2"
assignvariableop_55_adam_decay*
&assignvariableop_56_adam_learning_rate
assignvariableop_57_total
assignvariableop_58_count
assignvariableop_59_total_1
assignvariableop_60_count_1.
*assignvariableop_61_adam_conv2d_8_kernel_m,
(assignvariableop_62_adam_conv2d_8_bias_m.
*assignvariableop_63_adam_conv2d_9_kernel_m,
(assignvariableop_64_adam_conv2d_9_bias_m:
6assignvariableop_65_adam_batch_normalization_7_gamma_m9
5assignvariableop_66_adam_batch_normalization_7_beta_m/
+assignvariableop_67_adam_conv2d_10_kernel_m-
)assignvariableop_68_adam_conv2d_10_bias_m:
6assignvariableop_69_adam_batch_normalization_8_gamma_m9
5assignvariableop_70_adam_batch_normalization_8_beta_m/
+assignvariableop_71_adam_conv2d_11_kernel_m-
)assignvariableop_72_adam_conv2d_11_bias_m:
6assignvariableop_73_adam_batch_normalization_9_gamma_m9
5assignvariableop_74_adam_batch_normalization_9_beta_m/
+assignvariableop_75_adam_conv2d_12_kernel_m-
)assignvariableop_76_adam_conv2d_12_bias_m;
7assignvariableop_77_adam_batch_normalization_10_gamma_m:
6assignvariableop_78_adam_batch_normalization_10_beta_m/
+assignvariableop_79_adam_conv2d_13_kernel_m-
)assignvariableop_80_adam_conv2d_13_bias_m;
7assignvariableop_81_adam_batch_normalization_11_gamma_m:
6assignvariableop_82_adam_batch_normalization_11_beta_m/
+assignvariableop_83_adam_conv2d_14_kernel_m-
)assignvariableop_84_adam_conv2d_14_bias_m;
7assignvariableop_85_adam_batch_normalization_12_gamma_m:
6assignvariableop_86_adam_batch_normalization_12_beta_m/
+assignvariableop_87_adam_conv2d_15_kernel_m-
)assignvariableop_88_adam_conv2d_15_bias_m;
7assignvariableop_89_adam_batch_normalization_13_gamma_m:
6assignvariableop_90_adam_batch_normalization_13_beta_m-
)assignvariableop_91_adam_dense_4_kernel_m+
'assignvariableop_92_adam_dense_4_bias_m-
)assignvariableop_93_adam_dense_5_kernel_m+
'assignvariableop_94_adam_dense_5_bias_m-
)assignvariableop_95_adam_dense_6_kernel_m+
'assignvariableop_96_adam_dense_6_bias_m-
)assignvariableop_97_adam_dense_7_kernel_m+
'assignvariableop_98_adam_dense_7_bias_m.
*assignvariableop_99_adam_conv2d_8_kernel_v-
)assignvariableop_100_adam_conv2d_8_bias_v/
+assignvariableop_101_adam_conv2d_9_kernel_v-
)assignvariableop_102_adam_conv2d_9_bias_v;
7assignvariableop_103_adam_batch_normalization_7_gamma_v:
6assignvariableop_104_adam_batch_normalization_7_beta_v0
,assignvariableop_105_adam_conv2d_10_kernel_v.
*assignvariableop_106_adam_conv2d_10_bias_v;
7assignvariableop_107_adam_batch_normalization_8_gamma_v:
6assignvariableop_108_adam_batch_normalization_8_beta_v0
,assignvariableop_109_adam_conv2d_11_kernel_v.
*assignvariableop_110_adam_conv2d_11_bias_v;
7assignvariableop_111_adam_batch_normalization_9_gamma_v:
6assignvariableop_112_adam_batch_normalization_9_beta_v0
,assignvariableop_113_adam_conv2d_12_kernel_v.
*assignvariableop_114_adam_conv2d_12_bias_v<
8assignvariableop_115_adam_batch_normalization_10_gamma_v;
7assignvariableop_116_adam_batch_normalization_10_beta_v0
,assignvariableop_117_adam_conv2d_13_kernel_v.
*assignvariableop_118_adam_conv2d_13_bias_v<
8assignvariableop_119_adam_batch_normalization_11_gamma_v;
7assignvariableop_120_adam_batch_normalization_11_beta_v0
,assignvariableop_121_adam_conv2d_14_kernel_v.
*assignvariableop_122_adam_conv2d_14_bias_v<
8assignvariableop_123_adam_batch_normalization_12_gamma_v;
7assignvariableop_124_adam_batch_normalization_12_beta_v0
,assignvariableop_125_adam_conv2d_15_kernel_v.
*assignvariableop_126_adam_conv2d_15_bias_v<
8assignvariableop_127_adam_batch_normalization_13_gamma_v;
7assignvariableop_128_adam_batch_normalization_13_beta_v.
*assignvariableop_129_adam_dense_4_kernel_v,
(assignvariableop_130_adam_dense_4_bias_v.
*assignvariableop_131_adam_dense_5_kernel_v,
(assignvariableop_132_adam_dense_5_bias_v.
*assignvariableop_133_adam_dense_6_kernel_v,
(assignvariableop_134_adam_dense_6_bias_v.
*assignvariableop_135_adam_dense_7_kernel_v,
(assignvariableop_136_adam_dense_7_bias_v
identity_138¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99ĶM
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ŲL
valueĪLBĖLB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Ŗ
value BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1„
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3„
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_7_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_7_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_7_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_7_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ø
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_10_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_10_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_8_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_8_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12½
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_8_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Į
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_8_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_11_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ŗ
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_11_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16·
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_9_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_9_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18½
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_9_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Į
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_9_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¬
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_12_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ŗ
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_12_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ø
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_10_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_10_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¾
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_10_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ā
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_10_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¬
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ŗ
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ø
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_11_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_11_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¾
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_11_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ā
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_11_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¬
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_14_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ŗ
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_14_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ø
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_12_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35·
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_12_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¾
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_12_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ā
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_12_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¬
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_15_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ŗ
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_15_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ø
AssignVariableOp_40AssignVariableOp0assignvariableop_40_batch_normalization_13_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_13_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¾
AssignVariableOp_42AssignVariableOp6assignvariableop_42_batch_normalization_13_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ā
AssignVariableOp_43AssignVariableOp:assignvariableop_43_batch_normalization_13_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ŗ
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_4_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ø
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_4_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ŗ
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_5_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ø
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_5_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ŗ
AssignVariableOp_48AssignVariableOp"assignvariableop_48_dense_6_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ø
AssignVariableOp_49AssignVariableOp assignvariableop_49_dense_6_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ŗ
AssignVariableOp_50AssignVariableOp"assignvariableop_50_dense_7_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ø
AssignVariableOp_51AssignVariableOp assignvariableop_51_dense_7_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_52„
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_iterIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53§
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_beta_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54§
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_beta_2Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¦
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_decayIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56®
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_learning_rateIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57”
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58”
AssignVariableOp_58AssignVariableOpassignvariableop_58_countIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59£
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60£
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_8_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62°
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_8_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63²
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_9_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64°
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_9_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¾
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_7_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66½
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_7_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67³
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_10_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68±
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_10_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¾
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_8_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70½
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_8_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71³
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_11_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_11_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¾
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_9_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74½
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_9_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75³
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_12_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76±
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_12_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77æ
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_10_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78¾
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_10_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79³
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv2d_13_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80±
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv2d_13_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81æ
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_11_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82¾
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_11_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83³
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_14_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84±
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_14_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85æ
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_batch_normalization_12_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86¾
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_batch_normalization_12_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87³
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_conv2d_15_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88±
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_conv2d_15_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89æ
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_13_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90¾
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_13_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91±
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_dense_4_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Æ
AssignVariableOp_92AssignVariableOp'assignvariableop_92_adam_dense_4_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93±
AssignVariableOp_93AssignVariableOp)assignvariableop_93_adam_dense_5_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Æ
AssignVariableOp_94AssignVariableOp'assignvariableop_94_adam_dense_5_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95±
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_dense_6_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96Æ
AssignVariableOp_96AssignVariableOp'assignvariableop_96_adam_dense_6_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97±
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_dense_7_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Æ
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_dense_7_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99²
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_conv2d_8_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100“
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_conv2d_8_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101¶
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_conv2d_9_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102“
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_conv2d_9_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103Ā
AssignVariableOp_103AssignVariableOp7assignvariableop_103_adam_batch_normalization_7_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104Į
AssignVariableOp_104AssignVariableOp6assignvariableop_104_adam_batch_normalization_7_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105·
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_10_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106µ
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_10_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107Ā
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adam_batch_normalization_8_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108Į
AssignVariableOp_108AssignVariableOp6assignvariableop_108_adam_batch_normalization_8_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109·
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_11_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110µ
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_11_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111Ā
AssignVariableOp_111AssignVariableOp7assignvariableop_111_adam_batch_normalization_9_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112Į
AssignVariableOp_112AssignVariableOp6assignvariableop_112_adam_batch_normalization_9_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113·
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_12_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114µ
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_12_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115Ć
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_10_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116Ā
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_10_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117·
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_13_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118µ
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_13_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119Ć
AssignVariableOp_119AssignVariableOp8assignvariableop_119_adam_batch_normalization_11_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120Ā
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adam_batch_normalization_11_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121·
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv2d_14_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122µ
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv2d_14_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123Ć
AssignVariableOp_123AssignVariableOp8assignvariableop_123_adam_batch_normalization_12_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124Ā
AssignVariableOp_124AssignVariableOp7assignvariableop_124_adam_batch_normalization_12_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125·
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv2d_15_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126µ
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv2d_15_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127Ć
AssignVariableOp_127AssignVariableOp8assignvariableop_127_adam_batch_normalization_13_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128Ā
AssignVariableOp_128AssignVariableOp7assignvariableop_128_adam_batch_normalization_13_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129µ
AssignVariableOp_129AssignVariableOp*assignvariableop_129_adam_dense_4_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130³
AssignVariableOp_130AssignVariableOp(assignvariableop_130_adam_dense_4_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131µ
AssignVariableOp_131AssignVariableOp*assignvariableop_131_adam_dense_5_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132³
AssignVariableOp_132AssignVariableOp(assignvariableop_132_adam_dense_5_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133µ
AssignVariableOp_133AssignVariableOp*assignvariableop_133_adam_dense_6_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134³
AssignVariableOp_134AssignVariableOp(assignvariableop_134_adam_dense_6_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135µ
AssignVariableOp_135AssignVariableOp*assignvariableop_135_adam_dense_7_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136³
AssignVariableOp_136AssignVariableOp(assignvariableop_136_adam_dense_7_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpĖ
Identity_137Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_137æ
Identity_138IdentityIdentity_137:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_138"%
identity_138Identity_138:output:0*»
_input_shapes©
¦: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ģ
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_237799

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_237337

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240140

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239817

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_6_layer_call_fn_236736

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2367302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ø
Ŗ
7__inference_batch_normalization_12_layer_call_fn_240443

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2368292
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ų
©
6__inference_batch_normalization_7_layer_call_fn_239609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_237927

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_15_layer_call_and_return_conditional_losses_237697

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
®
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_236242

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
±
«
C__inference_dense_5_layer_call_and_return_conditional_losses_240751

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

®
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239965

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
«
C__inference_dense_6_layer_call_and_return_conditional_losses_240798

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_237519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_13_layer_call_and_return_conditional_losses_237466

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_240026

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_14_layer_call_and_return_conditional_losses_237597

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ž
Ŗ
7__inference_batch_normalization_11_layer_call_fn_240319

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2375012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē
c
*__inference_dropout_8_layer_call_fn_240031

inputs
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_237989

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü
©
6__inference_batch_normalization_8_layer_call_fn_239784

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
Ŗ
7__inference_batch_normalization_10_layer_call_fn_240120

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2374192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ŗ
7__inference_batch_normalization_13_layer_call_fn_240578

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2369022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ž
©
6__inference_batch_normalization_9_layer_call_fn_240009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß
}
(__inference_dense_7_layer_call_fn_240854

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2380132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
į
}
(__inference_dense_5_layer_call_fn_240760

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2378992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
©
6__inference_batch_normalization_9_layer_call_fn_239945

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2364932
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ķ
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_237804

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³
«
C__inference_dense_7_layer_call_and_return_conditional_losses_238013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240417

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ć
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_237106

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

®
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_237170

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
„

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239835

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
»
F
*__inference_dropout_9_layer_call_fn_240359

inputs
identityĻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375732
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
«
C__inference_dense_5_layer_call_and_return_conditional_losses_237899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239983

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240399

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ē
c
*__inference_dropout_9_layer_call_fn_240354

inputs
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
~
)__inference_conv2d_9_layer_call_fn_239558

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2370042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’..@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
Ø
Ŗ
7__inference_batch_normalization_11_layer_call_fn_240268

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2367132
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_14_layer_call_and_return_conditional_losses_240370

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_236713

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


*__inference_conv2d_11_layer_call_fn_239881

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2372352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_15_layer_call_and_return_conditional_losses_240518

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_237188

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_240725

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
«
C__inference_dense_6_layer_call_and_return_conditional_losses_237956

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


*__inference_conv2d_10_layer_call_fn_239733

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2371352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


*__inference_conv2d_12_layer_call_fn_240056

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2373662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ž
Ŗ
7__inference_batch_normalization_12_layer_call_fn_240494

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2376322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240463

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


*__inference_conv2d_14_layer_call_fn_240379

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2375972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ž
Ŗ
7__inference_batch_normalization_13_layer_call_fn_240642

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2377322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ø
Ŗ
7__inference_batch_normalization_13_layer_call_fn_240591

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2369332
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
õ

$__inference_signature_wrapper_238815
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_2361802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input
Ö
Æ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_236902

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
į
}
(__inference_dense_6_layer_call_fn_240807

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2379562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
śĶ
Ż
H__inference_sequential_1_layer_call_and_return_conditional_losses_239288

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpæ
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_8/Relu°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_9/Conv2D/ReadVariableOpÓ
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingSAME*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_9/Relu¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpļ
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ć
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3Ö
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool
dropout_7/IdentityIdentity max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_7/Identity“
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_10/Conv2D/ReadVariableOp×
conv2d_10/Conv2DConv2Ddropout_7/Identity:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_10/Conv2D«
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp±
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_10/Relu·
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOp½
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1ź
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpš
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1é
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3µ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpę
conv2d_11/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_11/Conv2D«
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp±
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_11/Relu·
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_9/ReadVariableOp½
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1ź
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpš
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1é
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3×
max_pooling2d_5/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool
dropout_8/IdentityIdentity max_pooling2d_5/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_8/Identityµ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp×
conv2d_12/Conv2DConv2Ddropout_8/Identity:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_12/Conv2D«
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp±
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_12/BiasAdd
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_12/Reluŗ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_10/ReadVariableOpĄ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1ķ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ļ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3µ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOpē
conv2d_13/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_13/Reluŗ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpĄ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1ķ
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ļ
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3Ų
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool
dropout_9/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_9/Identityµ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOp×
conv2d_14/Conv2DConv2Ddropout_9/Identity:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_14/Conv2D«
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp±
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_14/Reluŗ
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_12/ReadVariableOpĄ
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1ķ
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ļ
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3µ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpē
conv2d_15/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_15/Conv2D«
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp±
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_15/BiasAdd
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_15/Reluŗ
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpĄ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1ķ
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ļ
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3Ų
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool
dropout_10/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_10/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten_1/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/MatMul„
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/Relu
dropout_11/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/Identity§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_5/MatMul/ReadVariableOp¢
dense_5/MatMulMatMuldropout_11/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/MatMul„
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¢
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/Relu
dropout_12/IdentityIdentitydense_5/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_12/Identity§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_6/MatMul/ReadVariableOp¢
dense_6/MatMulMatMuldropout_12/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/MatMul„
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/Relu
dropout_13/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_13/Identity¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_7/MatMul/ReadVariableOp”
dense_7/MatMulMatMuldropout_13/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp”
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/SoftmaxÖ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulm
IdentityIdentitydense_7/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00:::::::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
Š

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@:::::W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239660

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:::::i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240094

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤
©
6__inference_batch_normalization_9_layer_call_fn_239932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2364622
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ļ
Ū=
__inference__traced_save_241299
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableop4
0savev2_adam_conv2d_14_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_m_read_readvariableop6
2savev2_adam_conv2d_15_kernel_m_read_readvariableop4
0savev2_adam_conv2d_15_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableop4
0savev2_adam_conv2d_14_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_v_read_readvariableop6
2savev2_adam_conv2d_15_kernel_v_read_readvariableop4
0savev2_adam_conv2d_15_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7160241deb0b41b2b665ddb3786933ce/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĒM
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ŲL
valueĪLBĖLB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names”
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Ŗ
value BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices;
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop>savev2_adam_batch_normalization_10_gamma_m_read_readvariableop=savev2_adam_batch_normalization_10_beta_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop0savev2_adam_conv2d_14_bias_m_read_readvariableop>savev2_adam_batch_normalization_12_gamma_m_read_readvariableop=savev2_adam_batch_normalization_12_beta_m_read_readvariableop2savev2_adam_conv2d_15_kernel_m_read_readvariableop0savev2_adam_conv2d_15_bias_m_read_readvariableop>savev2_adam_batch_normalization_13_gamma_m_read_readvariableop=savev2_adam_batch_normalization_13_beta_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop>savev2_adam_batch_normalization_10_gamma_v_read_readvariableop=savev2_adam_batch_normalization_10_beta_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop0savev2_adam_conv2d_14_bias_v_read_readvariableop>savev2_adam_batch_normalization_12_gamma_v_read_readvariableop=savev2_adam_batch_normalization_12_beta_v_read_readvariableop2savev2_adam_conv2d_15_kernel_v_read_readvariableop0savev2_adam_conv2d_15_bias_v_read_readvariableop>savev2_adam_batch_normalization_13_gamma_v_read_readvariableop=savev2_adam_batch_normalization_13_beta_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ü	
_input_shapesź	
ē	: :@:@:@@:@:@:@:@:@:@::::::::::::::::::::::::::::::::::::
::
::
::	:: : : : : : : : : :@:@:@@:@:@:@:@::::::::::::::::::::::::
::
::
::	::@:@:@@:@:@:@:@::::::::::::::::::::::::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-	)
'
_output_shapes
:@:!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::.'*
(
_output_shapes
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::!+

_output_shapes	
::!,

_output_shapes	
::&-"
 
_output_shapes
:
:!.

_output_shapes	
::&/"
 
_output_shapes
:
:!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::%3!

_output_shapes
:	: 4

_output_shapes
::5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :,>(
&
_output_shapes
:@: ?

_output_shapes
:@:,@(
&
_output_shapes
:@@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:-D)
'
_output_shapes
:@:!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::.H*
(
_output_shapes
::!I

_output_shapes	
::!J

_output_shapes	
::!K

_output_shapes	
::.L*
(
_output_shapes
::!M

_output_shapes	
::!N

_output_shapes	
::!O

_output_shapes	
::.P*
(
_output_shapes
::!Q

_output_shapes	
::!R

_output_shapes	
::!S

_output_shapes	
::.T*
(
_output_shapes
::!U

_output_shapes	
::!V

_output_shapes	
::!W

_output_shapes	
::.X*
(
_output_shapes
::!Y

_output_shapes	
::!Z

_output_shapes	
::![

_output_shapes	
::&\"
 
_output_shapes
:
:!]

_output_shapes	
::&^"
 
_output_shapes
:
:!_

_output_shapes	
::&`"
 
_output_shapes
:
:!a

_output_shapes	
::%b!

_output_shapes
:	: c

_output_shapes
::,d(
&
_output_shapes
:@: e

_output_shapes
:@:,f(
&
_output_shapes
:@@: g

_output_shapes
:@: h

_output_shapes
:@: i

_output_shapes
:@:-j)
'
_output_shapes
:@:!k

_output_shapes	
::!l

_output_shapes	
::!m

_output_shapes	
::.n*
(
_output_shapes
::!o

_output_shapes	
::!p

_output_shapes	
::!q

_output_shapes	
::.r*
(
_output_shapes
::!s

_output_shapes	
::!t

_output_shapes	
::!u

_output_shapes	
::.v*
(
_output_shapes
::!w

_output_shapes	
::!x

_output_shapes	
::!y

_output_shapes	
::.z*
(
_output_shapes
::!{

_output_shapes	
::!|

_output_shapes	
::!}

_output_shapes	
::.~*
(
_output_shapes
::!

_output_shapes	
::"

_output_shapes	
::"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::'"
 
_output_shapes
:
:"

_output_shapes	
::&!

_output_shapes
:	:!

_output_shapes
::

_output_shapes
: 
Ć
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_239698

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


*__inference_conv2d_13_layer_call_fn_240204

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2374662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾ 
­
H__inference_sequential_1_layer_call_and_return_conditional_losses_238181
conv2d_8_input
conv2d_8_238039
conv2d_8_238041
conv2d_9_238044
conv2d_9_238046 
batch_normalization_7_238049 
batch_normalization_7_238051 
batch_normalization_7_238053 
batch_normalization_7_238055
conv2d_10_238060
conv2d_10_238062 
batch_normalization_8_238065 
batch_normalization_8_238067 
batch_normalization_8_238069 
batch_normalization_8_238071
conv2d_11_238074
conv2d_11_238076 
batch_normalization_9_238079 
batch_normalization_9_238081 
batch_normalization_9_238083 
batch_normalization_9_238085
conv2d_12_238090
conv2d_12_238092!
batch_normalization_10_238095!
batch_normalization_10_238097!
batch_normalization_10_238099!
batch_normalization_10_238101
conv2d_13_238104
conv2d_13_238106!
batch_normalization_11_238109!
batch_normalization_11_238111!
batch_normalization_11_238113!
batch_normalization_11_238115
conv2d_14_238120
conv2d_14_238122!
batch_normalization_12_238125!
batch_normalization_12_238127!
batch_normalization_12_238129!
batch_normalization_12_238131
conv2d_15_238134
conv2d_15_238136!
batch_normalization_13_238139!
batch_normalization_13_238141!
batch_normalization_13_238143!
batch_normalization_13_238145
dense_4_238151
dense_4_238153
dense_5_238157
dense_5_238159
dense_6_238163
dense_6_238165
dense_7_238169
dense_7_238171
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall§
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_238039conv2d_8_238041*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2369772"
 conv2d_8/StatefulPartitionedCallĀ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_238044conv2d_9_238046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2370042"
 conv2d_9/StatefulPartitionedCallĆ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_7_238049batch_normalization_7_238051batch_normalization_7_238053batch_normalization_7_238055*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370572/
-batch_normalization_7/StatefulPartitionedCall¤
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_2362902!
max_pooling2d_4/PartitionedCall
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371112
dropout_7/PartitionedCallĮ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_10_238060conv2d_10_238062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2371352#
!conv2d_10/StatefulPartitionedCallÅ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_8_238065batch_normalization_8_238067batch_normalization_8_238069batch_normalization_8_238071*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371882/
-batch_normalization_8/StatefulPartitionedCallÕ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_11_238074conv2d_11_238076*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2372352#
!conv2d_11/StatefulPartitionedCallÅ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_9_238079batch_normalization_9_238081batch_normalization_9_238083batch_normalization_9_238085*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372882/
-batch_normalization_9/StatefulPartitionedCall„
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2365102!
max_pooling2d_5/PartitionedCall
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373422
dropout_8/PartitionedCallĮ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_12_238090conv2d_12_238092*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2373662#
!conv2d_12/StatefulPartitionedCallĢ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_10_238095batch_normalization_10_238097batch_normalization_10_238099batch_normalization_10_238101*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23741920
.batch_normalization_10/StatefulPartitionedCallÖ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_13_238104conv2d_13_238106*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2374662#
!conv2d_13/StatefulPartitionedCallĢ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_11_238109batch_normalization_11_238111batch_normalization_11_238113batch_normalization_11_238115*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23751920
.batch_normalization_11/StatefulPartitionedCall¦
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2367302!
max_pooling2d_6/PartitionedCall
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375732
dropout_9/PartitionedCallĮ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_14_238120conv2d_14_238122*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2375972#
!conv2d_14/StatefulPartitionedCallĢ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_12_238125batch_normalization_12_238127batch_normalization_12_238129batch_normalization_12_238131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_23765020
.batch_normalization_12/StatefulPartitionedCallÖ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_15_238134conv2d_15_238136*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2376972#
!conv2d_15/StatefulPartitionedCallĢ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_13_238139batch_normalization_13_238141batch_normalization_13_238143batch_normalization_13_238145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_23775020
.batch_normalization_13/StatefulPartitionedCall¦
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2369502!
max_pooling2d_7/PartitionedCall
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2378042
dropout_10/PartitionedCallų
flatten_1/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2378232
flatten_1/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_238151dense_4_238153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2378422!
dense_4/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378752
dropout_11/PartitionedCall°
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_238157dense_5_238159*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2378992!
dense_5/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379322
dropout_12/PartitionedCall°
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_6_238163dense_6_238165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2379562!
dense_6/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379892
dropout_13/PartitionedCallÆ
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_7_238169dense_7_238171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2380132!
dense_7/StatefulPartitionedCall¾
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_238039*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulö
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input

®
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_237039

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_240730

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240547

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
©
d
+__inference_dropout_12_layer_call_fn_240782

inputs
identity¢StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ļ
¬
D__inference_conv2d_8_layer_call_and_return_conditional_losses_239529

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
ReluĶ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’00:::W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_237632

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

®
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_237270

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_236950

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ģ
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_240667

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_236510

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_237932

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_237568

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240288

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_236358

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_236829

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ü

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
Ŗ
7__inference_batch_normalization_13_layer_call_fn_240655

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2377502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

G
+__inference_dropout_13_layer_call_fn_240834

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_11_layer_call_and_return_conditional_losses_239872

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_12_layer_call_and_return_conditional_losses_237366

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_10_layer_call_and_return_conditional_losses_237135

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’@:::W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_236682

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


-__inference_sequential_1_layer_call_fn_239506

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2385832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240629

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
d
+__inference_dropout_10_layer_call_fn_240677

inputs
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2377992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_236462

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_236730

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239901

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_236798

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ė«
­
H__inference_sequential_1_layer_call_and_return_conditional_losses_238036
conv2d_8_input
conv2d_8_236988
conv2d_8_236990
conv2d_9_237015
conv2d_9_237017 
batch_normalization_7_237084 
batch_normalization_7_237086 
batch_normalization_7_237088 
batch_normalization_7_237090
conv2d_10_237146
conv2d_10_237148 
batch_normalization_8_237215 
batch_normalization_8_237217 
batch_normalization_8_237219 
batch_normalization_8_237221
conv2d_11_237246
conv2d_11_237248 
batch_normalization_9_237315 
batch_normalization_9_237317 
batch_normalization_9_237319 
batch_normalization_9_237321
conv2d_12_237377
conv2d_12_237379!
batch_normalization_10_237446!
batch_normalization_10_237448!
batch_normalization_10_237450!
batch_normalization_10_237452
conv2d_13_237477
conv2d_13_237479!
batch_normalization_11_237546!
batch_normalization_11_237548!
batch_normalization_11_237550!
batch_normalization_11_237552
conv2d_14_237608
conv2d_14_237610!
batch_normalization_12_237677!
batch_normalization_12_237679!
batch_normalization_12_237681!
batch_normalization_12_237683
conv2d_15_237708
conv2d_15_237710!
batch_normalization_13_237777!
batch_normalization_13_237779!
batch_normalization_13_237781!
batch_normalization_13_237783
dense_4_237853
dense_4_237855
dense_5_237910
dense_5_237912
dense_6_237967
dense_6_237969
dense_7_238024
dense_7_238026
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall§
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_236988conv2d_8_236990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2369772"
 conv2d_8/StatefulPartitionedCallĀ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_237015conv2d_9_237017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2370042"
 conv2d_9/StatefulPartitionedCallĮ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_7_237084batch_normalization_7_237086batch_normalization_7_237088batch_normalization_7_237090*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370392/
-batch_normalization_7/StatefulPartitionedCall¤
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_2362902!
max_pooling2d_4/PartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371062#
!dropout_7/StatefulPartitionedCallÉ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_10_237146conv2d_10_237148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2371352#
!conv2d_10/StatefulPartitionedCallĆ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_8_237215batch_normalization_8_237217batch_normalization_8_237219batch_normalization_8_237221*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371702/
-batch_normalization_8/StatefulPartitionedCallÕ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_11_237246conv2d_11_237248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2372352#
!conv2d_11/StatefulPartitionedCallĆ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_9_237315batch_normalization_9_237317batch_normalization_9_237319batch_normalization_9_237321*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372702/
-batch_normalization_9/StatefulPartitionedCall„
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2365102!
max_pooling2d_5/PartitionedCallĮ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373372#
!dropout_8/StatefulPartitionedCallÉ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_12_237377conv2d_12_237379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2373662#
!conv2d_12/StatefulPartitionedCallŹ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_10_237446batch_normalization_10_237448batch_normalization_10_237450batch_normalization_10_237452*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23740120
.batch_normalization_10/StatefulPartitionedCallÖ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_13_237477conv2d_13_237479*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2374662#
!conv2d_13/StatefulPartitionedCallŹ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_11_237546batch_normalization_11_237548batch_normalization_11_237550batch_normalization_11_237552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23750120
.batch_normalization_11/StatefulPartitionedCall¦
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2367302!
max_pooling2d_6/PartitionedCallĮ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375682#
!dropout_9/StatefulPartitionedCallÉ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_14_237608conv2d_14_237610*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2375972#
!conv2d_14/StatefulPartitionedCallŹ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_12_237677batch_normalization_12_237679batch_normalization_12_237681batch_normalization_12_237683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_23763220
.batch_normalization_12/StatefulPartitionedCallÖ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_15_237708conv2d_15_237710*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2376972#
!conv2d_15/StatefulPartitionedCallŹ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_13_237777batch_normalization_13_237779batch_normalization_13_237781batch_normalization_13_237783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_23773220
.batch_normalization_13/StatefulPartitionedCall¦
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2369502!
max_pooling2d_7/PartitionedCallÄ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2377992$
"dropout_10/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2378232
flatten_1/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_237853dense_4_237855*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2378422!
dense_4/StatefulPartitionedCall½
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378702$
"dropout_11/StatefulPartitionedCallø
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_237910dense_5_237912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2378992!
dense_5/StatefulPartitionedCall½
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379272$
"dropout_12/StatefulPartitionedCallø
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_6_237967dense_6_237969*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2379562!
dense_6/StatefulPartitionedCall½
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379842$
"dropout_13/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_7_238024dense_7_238026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2380132!
dense_7/StatefulPartitionedCall¾
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_236988*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulö
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_237984

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ęš
ā
!__inference__wrapped_model_236180
conv2d_8_input8
4sequential_1_conv2d_8_conv2d_readvariableop_resource9
5sequential_1_conv2d_8_biasadd_readvariableop_resource8
4sequential_1_conv2d_9_conv2d_readvariableop_resource9
5sequential_1_conv2d_9_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_7_readvariableop_resource@
<sequential_1_batch_normalization_7_readvariableop_1_resourceO
Ksequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_10_conv2d_readvariableop_resource:
6sequential_1_conv2d_10_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_8_readvariableop_resource@
<sequential_1_batch_normalization_8_readvariableop_1_resourceO
Ksequential_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_11_conv2d_readvariableop_resource:
6sequential_1_conv2d_11_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_9_readvariableop_resource@
<sequential_1_batch_normalization_9_readvariableop_1_resourceO
Ksequential_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_12_conv2d_readvariableop_resource:
6sequential_1_conv2d_12_biasadd_readvariableop_resource?
;sequential_1_batch_normalization_10_readvariableop_resourceA
=sequential_1_batch_normalization_10_readvariableop_1_resourceP
Lsequential_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceR
Nsequential_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_13_conv2d_readvariableop_resource:
6sequential_1_conv2d_13_biasadd_readvariableop_resource?
;sequential_1_batch_normalization_11_readvariableop_resourceA
=sequential_1_batch_normalization_11_readvariableop_1_resourceP
Lsequential_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceR
Nsequential_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_14_conv2d_readvariableop_resource:
6sequential_1_conv2d_14_biasadd_readvariableop_resource?
;sequential_1_batch_normalization_12_readvariableop_resourceA
=sequential_1_batch_normalization_12_readvariableop_1_resourceP
Lsequential_1_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceR
Nsequential_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource9
5sequential_1_conv2d_15_conv2d_readvariableop_resource:
6sequential_1_conv2d_15_biasadd_readvariableop_resource?
;sequential_1_batch_normalization_13_readvariableop_resourceA
=sequential_1_batch_normalization_13_readvariableop_1_resourceP
Lsequential_1_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceR
Nsequential_1_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource7
3sequential_1_dense_4_matmul_readvariableop_resource8
4sequential_1_dense_4_biasadd_readvariableop_resource7
3sequential_1_dense_5_matmul_readvariableop_resource8
4sequential_1_dense_5_biasadd_readvariableop_resource7
3sequential_1_dense_6_matmul_readvariableop_resource8
4sequential_1_dense_6_biasadd_readvariableop_resource7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource
identity×
+sequential_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+sequential_1/conv2d_8/Conv2D/ReadVariableOpī
sequential_1/conv2d_8/Conv2DConv2Dconv2d_8_input3sequential_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingVALID*
strides
2
sequential_1/conv2d_8/Conv2DĪ
,sequential_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/conv2d_8/BiasAdd/ReadVariableOpą
sequential_1/conv2d_8/BiasAddBiasAdd%sequential_1/conv2d_8/Conv2D:output:04sequential_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
sequential_1/conv2d_8/BiasAdd¢
sequential_1/conv2d_8/ReluRelu&sequential_1/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
sequential_1/conv2d_8/Relu×
+sequential_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+sequential_1/conv2d_9/Conv2D/ReadVariableOp
sequential_1/conv2d_9/Conv2DConv2D(sequential_1/conv2d_8/Relu:activations:03sequential_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingSAME*
strides
2
sequential_1/conv2d_9/Conv2DĪ
,sequential_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/conv2d_9/BiasAdd/ReadVariableOpą
sequential_1/conv2d_9/BiasAddBiasAdd%sequential_1/conv2d_9/Conv2D:output:04sequential_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
sequential_1/conv2d_9/BiasAdd¢
sequential_1/conv2d_9/ReluRelu&sequential_1/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
sequential_1/conv2d_9/ReluŻ
1sequential_1/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_1/batch_normalization_7/ReadVariableOpć
3sequential_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_1/batch_normalization_7/ReadVariableOp_1
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¾
3sequential_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(sequential_1/conv2d_9/Relu:activations:09sequential_1/batch_normalization_7/ReadVariableOp:value:0;sequential_1/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
is_training( 25
3sequential_1/batch_normalization_7/FusedBatchNormV3ż
$sequential_1/max_pooling2d_4/MaxPoolMaxPool7sequential_1/batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_4/MaxPool·
sequential_1/dropout_7/IdentityIdentity-sequential_1/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2!
sequential_1/dropout_7/IdentityŪ
,sequential_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02.
,sequential_1/conv2d_10/Conv2D/ReadVariableOp
sequential_1/conv2d_10/Conv2DConv2D(sequential_1/dropout_7/Identity:output:04sequential_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_10/Conv2DŅ
-sequential_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_10/BiasAdd/ReadVariableOpå
sequential_1/conv2d_10/BiasAddBiasAdd&sequential_1/conv2d_10/Conv2D:output:05sequential_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_10/BiasAdd¦
sequential_1/conv2d_10/ReluRelu'sequential_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_10/ReluŽ
1sequential_1/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_1/batch_normalization_8/ReadVariableOpä
3sequential_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_1/batch_normalization_8/ReadVariableOp_1
Bsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Dsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ä
3sequential_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_10/Relu:activations:09sequential_1/batch_normalization_8/ReadVariableOp:value:0;sequential_1/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 25
3sequential_1/batch_normalization_8/FusedBatchNormV3Ü
,sequential_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,sequential_1/conv2d_11/Conv2D/ReadVariableOp
sequential_1/conv2d_11/Conv2DConv2D7sequential_1/batch_normalization_8/FusedBatchNormV3:y:04sequential_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_11/Conv2DŅ
-sequential_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_11/BiasAdd/ReadVariableOpå
sequential_1/conv2d_11/BiasAddBiasAdd&sequential_1/conv2d_11/Conv2D:output:05sequential_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_11/BiasAdd¦
sequential_1/conv2d_11/ReluRelu'sequential_1/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_11/ReluŽ
1sequential_1/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_9_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_1/batch_normalization_9/ReadVariableOpä
3sequential_1/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_1/batch_normalization_9/ReadVariableOp_1
Bsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Dsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ä
3sequential_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_11/Relu:activations:09sequential_1/batch_normalization_9/ReadVariableOp:value:0;sequential_1/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 25
3sequential_1/batch_normalization_9/FusedBatchNormV3ž
$sequential_1/max_pooling2d_5/MaxPoolMaxPool7sequential_1/batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_5/MaxPoolø
sequential_1/dropout_8/IdentityIdentity-sequential_1/max_pooling2d_5/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
sequential_1/dropout_8/IdentityÜ
,sequential_1/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,sequential_1/conv2d_12/Conv2D/ReadVariableOp
sequential_1/conv2d_12/Conv2DConv2D(sequential_1/dropout_8/Identity:output:04sequential_1/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_12/Conv2DŅ
-sequential_1/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_12/BiasAdd/ReadVariableOpå
sequential_1/conv2d_12/BiasAddBiasAdd&sequential_1/conv2d_12/Conv2D:output:05sequential_1/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_12/BiasAdd¦
sequential_1/conv2d_12/ReluRelu'sequential_1/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_12/Reluį
2sequential_1/batch_normalization_10/ReadVariableOpReadVariableOp;sequential_1_batch_normalization_10_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_1/batch_normalization_10/ReadVariableOpē
4sequential_1/batch_normalization_10/ReadVariableOp_1ReadVariableOp=sequential_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_1/batch_normalization_10/ReadVariableOp_1
Csequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Esequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ź
4sequential_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_12/Relu:activations:0:sequential_1/batch_normalization_10/ReadVariableOp:value:0<sequential_1/batch_normalization_10/ReadVariableOp_1:value:0Ksequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Msequential_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_1/batch_normalization_10/FusedBatchNormV3Ü
,sequential_1/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,sequential_1/conv2d_13/Conv2D/ReadVariableOp
sequential_1/conv2d_13/Conv2DConv2D8sequential_1/batch_normalization_10/FusedBatchNormV3:y:04sequential_1/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_13/Conv2DŅ
-sequential_1/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_13/BiasAdd/ReadVariableOpå
sequential_1/conv2d_13/BiasAddBiasAdd&sequential_1/conv2d_13/Conv2D:output:05sequential_1/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_13/BiasAdd¦
sequential_1/conv2d_13/ReluRelu'sequential_1/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_13/Reluį
2sequential_1/batch_normalization_11/ReadVariableOpReadVariableOp;sequential_1_batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_1/batch_normalization_11/ReadVariableOpē
4sequential_1/batch_normalization_11/ReadVariableOp_1ReadVariableOp=sequential_1_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_1/batch_normalization_11/ReadVariableOp_1
Csequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Esequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ź
4sequential_1/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_13/Relu:activations:0:sequential_1/batch_normalization_11/ReadVariableOp:value:0<sequential_1/batch_normalization_11/ReadVariableOp_1:value:0Ksequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Msequential_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_1/batch_normalization_11/FusedBatchNormV3’
$sequential_1/max_pooling2d_6/MaxPoolMaxPool8sequential_1/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_6/MaxPoolø
sequential_1/dropout_9/IdentityIdentity-sequential_1/max_pooling2d_6/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
sequential_1/dropout_9/IdentityÜ
,sequential_1/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,sequential_1/conv2d_14/Conv2D/ReadVariableOp
sequential_1/conv2d_14/Conv2DConv2D(sequential_1/dropout_9/Identity:output:04sequential_1/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_14/Conv2DŅ
-sequential_1/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_14/BiasAdd/ReadVariableOpå
sequential_1/conv2d_14/BiasAddBiasAdd&sequential_1/conv2d_14/Conv2D:output:05sequential_1/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_14/BiasAdd¦
sequential_1/conv2d_14/ReluRelu'sequential_1/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_14/Reluį
2sequential_1/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_1_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_1/batch_normalization_12/ReadVariableOpē
4sequential_1/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_1_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_1/batch_normalization_12/ReadVariableOp_1
Csequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_1_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
Esequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ź
4sequential_1/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_14/Relu:activations:0:sequential_1/batch_normalization_12/ReadVariableOp:value:0<sequential_1/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_1/batch_normalization_12/FusedBatchNormV3Ü
,sequential_1/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_1_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,sequential_1/conv2d_15/Conv2D/ReadVariableOp
sequential_1/conv2d_15/Conv2DConv2D8sequential_1/batch_normalization_12/FusedBatchNormV3:y:04sequential_1/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
sequential_1/conv2d_15/Conv2DŅ
-sequential_1/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_1/conv2d_15/BiasAdd/ReadVariableOpå
sequential_1/conv2d_15/BiasAddBiasAdd&sequential_1/conv2d_15/Conv2D:output:05sequential_1/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2 
sequential_1/conv2d_15/BiasAdd¦
sequential_1/conv2d_15/ReluRelu'sequential_1/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
sequential_1/conv2d_15/Reluį
2sequential_1/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_1_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_1/batch_normalization_13/ReadVariableOpē
4sequential_1/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_1_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_1/batch_normalization_13/ReadVariableOp_1
Csequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_1_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
Esequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_1_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ź
4sequential_1/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3)sequential_1/conv2d_15/Relu:activations:0:sequential_1/batch_normalization_13/ReadVariableOp:value:0<sequential_1/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_1/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_1/batch_normalization_13/FusedBatchNormV3’
$sequential_1/max_pooling2d_7/MaxPoolMaxPool8sequential_1/batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_7/MaxPoolŗ
 sequential_1/dropout_10/IdentityIdentity-sequential_1/max_pooling2d_7/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2"
 sequential_1/dropout_10/Identity
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
sequential_1/flatten_1/ConstŠ
sequential_1/flatten_1/ReshapeReshape)sequential_1/dropout_10/Identity:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2 
sequential_1/flatten_1/ReshapeĪ
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOpŌ
sequential_1/dense_4/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_4/MatMulĢ
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOpÖ
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_4/BiasAdd
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_4/Relu¬
 sequential_1/dropout_11/IdentityIdentity'sequential_1/dense_4/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2"
 sequential_1/dropout_11/IdentityĪ
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOpÖ
sequential_1/dense_5/MatMulMatMul)sequential_1/dropout_11/Identity:output:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_5/MatMulĢ
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOpÖ
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_5/BiasAdd
sequential_1/dense_5/ReluRelu%sequential_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_5/Relu¬
 sequential_1/dropout_12/IdentityIdentity'sequential_1/dense_5/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2"
 sequential_1/dropout_12/IdentityĪ
*sequential_1/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*sequential_1/dense_6/MatMul/ReadVariableOpÖ
sequential_1/dense_6/MatMulMatMul)sequential_1/dropout_12/Identity:output:02sequential_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_6/MatMulĢ
+sequential_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_6/BiasAdd/ReadVariableOpÖ
sequential_1/dense_6/BiasAddBiasAdd%sequential_1/dense_6/MatMul:product:03sequential_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_6/BiasAdd
sequential_1/dense_6/ReluRelu%sequential_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_1/dense_6/Relu¬
 sequential_1/dropout_13/IdentityIdentity'sequential_1/dense_6/Relu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2"
 sequential_1/dropout_13/IdentityĶ
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOpÕ
sequential_1/dense_7/MatMulMatMul)sequential_1/dropout_13/Identity:output:02sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_1/dense_7/MatMulĖ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOpÕ
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_1/dense_7/BiasAdd 
sequential_1/dense_7/SoftmaxSoftmax%sequential_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_1/dense_7/Softmaxz
IdentityIdentity&sequential_1/dense_7/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00:::::::::::::::::::::::::::::::::::::::::::::::::::::_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input
Ć
c
*__inference_dropout_7_layer_call_fn_239708

inputs
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
	
­
E__inference_conv2d_11_layer_call_and_return_conditional_losses_237235

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_237057

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@:::::W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
±
«
C__inference_dense_4_layer_call_and_return_conditional_losses_237842

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_236609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_237501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240306

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_236273

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:::::i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_237732

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
’
~
)__inference_conv2d_8_layer_call_fn_239538

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2369772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’00::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs

Æ
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240611

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
 
©
6__inference_batch_normalization_7_layer_call_fn_239673

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2362422
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240158

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ü

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_237288

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_239703

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ś
©
6__inference_batch_normalization_7_layer_call_fn_239622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_237419

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240565

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
„

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_236493

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ž
Ŗ
7__inference_batch_normalization_10_layer_call_fn_240107

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2374012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_12_layer_call_and_return_conditional_losses_240047

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¢
©
6__inference_batch_normalization_7_layer_call_fn_239686

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2362732
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
į
}
(__inference_dense_4_layer_call_fn_240713

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2378422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_9_layer_call_and_return_conditional_losses_237004

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’..@:::W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs
©
d
+__inference_dropout_13_layer_call_fn_240829

inputs
identity¢StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

®
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:’’’’’’’’’..@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’..@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:’’’’’’’’’..@
 
_user_specified_nameinputs


*__inference_conv2d_15_layer_call_fn_240527

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2376972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
Ŗ
7__inference_batch_normalization_11_layer_call_fn_240332

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2375192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó«
„
H__inference_sequential_1_layer_call_and_return_conditional_losses_238329

inputs
conv2d_8_238187
conv2d_8_238189
conv2d_9_238192
conv2d_9_238194 
batch_normalization_7_238197 
batch_normalization_7_238199 
batch_normalization_7_238201 
batch_normalization_7_238203
conv2d_10_238208
conv2d_10_238210 
batch_normalization_8_238213 
batch_normalization_8_238215 
batch_normalization_8_238217 
batch_normalization_8_238219
conv2d_11_238222
conv2d_11_238224 
batch_normalization_9_238227 
batch_normalization_9_238229 
batch_normalization_9_238231 
batch_normalization_9_238233
conv2d_12_238238
conv2d_12_238240!
batch_normalization_10_238243!
batch_normalization_10_238245!
batch_normalization_10_238247!
batch_normalization_10_238249
conv2d_13_238252
conv2d_13_238254!
batch_normalization_11_238257!
batch_normalization_11_238259!
batch_normalization_11_238261!
batch_normalization_11_238263
conv2d_14_238268
conv2d_14_238270!
batch_normalization_12_238273!
batch_normalization_12_238275!
batch_normalization_12_238277!
batch_normalization_12_238279
conv2d_15_238282
conv2d_15_238284!
batch_normalization_13_238287!
batch_normalization_13_238289!
batch_normalization_13_238291!
batch_normalization_13_238293
dense_4_238299
dense_4_238301
dense_5_238305
dense_5_238307
dense_6_238311
dense_6_238313
dense_7_238317
dense_7_238319
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_238187conv2d_8_238189*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2369772"
 conv2d_8/StatefulPartitionedCallĀ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_238192conv2d_9_238194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2370042"
 conv2d_9/StatefulPartitionedCallĮ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_7_238197batch_normalization_7_238199batch_normalization_7_238201batch_normalization_7_238203*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370392/
-batch_normalization_7/StatefulPartitionedCall¤
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_2362902!
max_pooling2d_4/PartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371062#
!dropout_7/StatefulPartitionedCallÉ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_10_238208conv2d_10_238210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2371352#
!conv2d_10/StatefulPartitionedCallĆ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_8_238213batch_normalization_8_238215batch_normalization_8_238217batch_normalization_8_238219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371702/
-batch_normalization_8/StatefulPartitionedCallÕ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_11_238222conv2d_11_238224*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2372352#
!conv2d_11/StatefulPartitionedCallĆ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_9_238227batch_normalization_9_238229batch_normalization_9_238231batch_normalization_9_238233*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372702/
-batch_normalization_9/StatefulPartitionedCall„
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2365102!
max_pooling2d_5/PartitionedCallĮ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373372#
!dropout_8/StatefulPartitionedCallÉ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_12_238238conv2d_12_238240*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2373662#
!conv2d_12/StatefulPartitionedCallŹ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_10_238243batch_normalization_10_238245batch_normalization_10_238247batch_normalization_10_238249*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23740120
.batch_normalization_10/StatefulPartitionedCallÖ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_13_238252conv2d_13_238254*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2374662#
!conv2d_13/StatefulPartitionedCallŹ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_11_238257batch_normalization_11_238259batch_normalization_11_238261batch_normalization_11_238263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23750120
.batch_normalization_11/StatefulPartitionedCall¦
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2367302!
max_pooling2d_6/PartitionedCallĮ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375682#
!dropout_9/StatefulPartitionedCallÉ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_14_238268conv2d_14_238270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2375972#
!conv2d_14/StatefulPartitionedCallŹ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_12_238273batch_normalization_12_238275batch_normalization_12_238277batch_normalization_12_238279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_23763220
.batch_normalization_12/StatefulPartitionedCallÖ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_15_238282conv2d_15_238284*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2376972#
!conv2d_15/StatefulPartitionedCallŹ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_13_238287batch_normalization_13_238289batch_normalization_13_238291batch_normalization_13_238293*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_23773220
.batch_normalization_13/StatefulPartitionedCall¦
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2369502!
max_pooling2d_7/PartitionedCallÄ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2377992$
"dropout_10/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2378232
flatten_1/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_238299dense_4_238301*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2378422!
dense_4/StatefulPartitionedCall½
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378702$
"dropout_11/StatefulPartitionedCallø
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_238305dense_5_238307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2378992!
dense_5/StatefulPartitionedCall½
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379272$
"dropout_12/StatefulPartitionedCallø
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_6_238311dense_6_238313*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2379562!
dense_6/StatefulPartitionedCall½
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379842$
"dropout_13/StatefulPartitionedCall·
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_7_238317dense_7_238319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2380132!
dense_7/StatefulPartitionedCall¾
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_238187*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulö
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
Ü
©
6__inference_batch_normalization_9_layer_call_fn_239996

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ž
©
6__inference_batch_normalization_8_layer_call_fn_239797

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_240344

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_240819

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_236933

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ė
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_240021

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
Æ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240224

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_240772

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ŗ
7__inference_batch_normalization_11_layer_call_fn_240255

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2366822
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ģ
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_237342

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ŗ
7__inference_batch_normalization_12_layer_call_fn_240430

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2367982
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
«
F
*__inference_flatten_1_layer_call_fn_240693

inputs
identityĒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2378232
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_240688

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_240349

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
Ŗ
7__inference_batch_normalization_12_layer_call_fn_240507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2376502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_240824

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
­
E__inference_conv2d_13_layer_call_and_return_conditional_losses_240195

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’:::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

G
+__inference_dropout_12_layer_call_fn_240787

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_237870

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240481

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
„
”
-__inference_sequential_1_layer_call_fn_238690
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2385832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’00
(
_user_specified_nameconv2d_8_input
ŹĻ

H__inference_sequential_1_layer_call_and_return_conditional_losses_239083

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1¢%batch_normalization_12/AssignNewValue¢'batch_normalization_12/AssignNewValue_1¢%batch_normalization_13/AssignNewValue¢'batch_normalization_13/AssignNewValue_1¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpæ
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_8/Relu°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_9/Conv2D/ReadVariableOpÓ
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@*
paddingSAME*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’..@2
conv2d_9/Relu¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpļ
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ń
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_7/FusedBatchNormV3
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1Ö
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPoolw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/Const³
dropout_7/dropout/MulMul max_pooling2d_4/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_7/dropout/Mul
dropout_7/dropout/ShapeShape max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/ShapeŚ
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/yī
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2 
dropout_7/dropout/GreaterEqual„
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout_7/dropout/CastŖ
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_7/dropout/Mul_1“
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_10/Conv2D/ReadVariableOp×
conv2d_10/Conv2DConv2Ddropout_7/dropout/Mul_1:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_10/Conv2D«
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp±
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_10/Relu·
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOp½
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1ź
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpš
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1÷
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_8/FusedBatchNormV3
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1µ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpę
conv2d_11/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_11/Conv2D«
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp±
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_11/Relu·
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_9/ReadVariableOp½
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_9/ReadVariableOp_1ź
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpš
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1÷
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_9/FusedBatchNormV3
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1×
max_pooling2d_5/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_8/dropout/Const“
dropout_8/dropout/MulMul max_pooling2d_5/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_8/dropout/Mul
dropout_8/dropout/ShapeShape max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/ShapeŪ
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_8/dropout/GreaterEqual/yļ
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2 
dropout_8/dropout/GreaterEqual¦
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout_8/dropout/Cast«
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_8/dropout/Mul_1µ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp×
conv2d_12/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_12/Conv2D«
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp±
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_12/BiasAdd
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_12/Reluŗ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_10/ReadVariableOpĄ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1ķ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ż
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_10/FusedBatchNormV3
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1µ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOpē
conv2d_13/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_13/Reluŗ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpĄ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1ķ
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ż
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_11/FusedBatchNormV3
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1Ų
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const“
dropout_9/dropout/MulMul max_pooling2d_6/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/ShapeŪ
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/yļ
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2 
dropout_9/dropout/GreaterEqual¦
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout_9/dropout/Cast«
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_9/dropout/Mul_1µ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOp×
conv2d_14/Conv2DConv2Ddropout_9/dropout/Mul_1:z:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_14/Conv2D«
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp±
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_14/Reluŗ
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_12/ReadVariableOpĄ
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_12/ReadVariableOp_1ķ
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ż
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_12/FusedBatchNormV3
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1µ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpē
conv2d_15/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2
conv2d_15/Conv2D«
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp±
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_15/BiasAdd
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
conv2d_15/Reluŗ
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpĄ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1ķ
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ż
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_13/FusedBatchNormV3
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1Ų
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPooly
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_10/dropout/Const·
dropout_10/dropout/MulMul max_pooling2d_7/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/ShapeŽ
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_10/dropout/GreaterEqual/yó
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
dropout_10/dropout/GreaterEqual©
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/CastÆ
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_10/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten_1/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/MatMul„
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_11/dropout/Const©
dropout_11/dropout/MulMuldense_4/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Mul~
dropout_11/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/ShapeÖ
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2#
!dropout_11/dropout/GreaterEqual/yė
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2!
dropout_11/dropout/GreaterEqual”
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Cast§
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Mul_1§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_5/MatMul/ReadVariableOp¢
dense_5/MatMulMatMuldropout_11/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/MatMul„
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¢
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_5/Reluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_12/dropout/Const©
dropout_12/dropout/MulMuldense_5/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_12/dropout/Mul~
dropout_12/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/ShapeÖ
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢĢ>2#
!dropout_12/dropout/GreaterEqual/yė
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2!
dropout_12/dropout/GreaterEqual”
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout_12/dropout/Cast§
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_12/dropout/Mul_1§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_6/MatMul/ReadVariableOp¢
dense_6/MatMulMatMuldropout_12/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/MatMul„
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_6/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Const©
dropout_13/dropout/MulMuldense_6/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_13/dropout/Mul~
dropout_13/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/ShapeÖ
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yė
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2!
dropout_13/dropout/GreaterEqual”
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout_13/dropout/Cast§
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_13/dropout/Mul_1¦
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_7/MatMul/ReadVariableOp”
dense_7/MatMulMatMuldropout_13/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp”
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_7/SoftmaxÖ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mul„
IdentityIdentitydense_7/Softmax:softmax:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
ņ

o
__inference_loss_fn_0_240865>
:conv2d_8_kernel_regularizer_square_readvariableop_resource
identityé
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_8_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulf
IdentityIdentity#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ķ
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_240672

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
½
G
+__inference_dropout_10_layer_call_fn_240682

inputs
identityŠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2378042
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_240777

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ø
Ŗ
7__inference_batch_normalization_10_layer_call_fn_240184

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2366092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
¦
Ŗ
7__inference_batch_normalization_10_layer_call_fn_240171

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2365782
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

®
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239753

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3’
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤
©
6__inference_batch_normalization_8_layer_call_fn_239848

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2363582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,’’’’’’’’’’’’’’’’’’’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż

R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_237750

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’:::::X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦ 
„
H__inference_sequential_1_layer_call_and_return_conditional_losses_238583

inputs
conv2d_8_238441
conv2d_8_238443
conv2d_9_238446
conv2d_9_238448 
batch_normalization_7_238451 
batch_normalization_7_238453 
batch_normalization_7_238455 
batch_normalization_7_238457
conv2d_10_238462
conv2d_10_238464 
batch_normalization_8_238467 
batch_normalization_8_238469 
batch_normalization_8_238471 
batch_normalization_8_238473
conv2d_11_238476
conv2d_11_238478 
batch_normalization_9_238481 
batch_normalization_9_238483 
batch_normalization_9_238485 
batch_normalization_9_238487
conv2d_12_238492
conv2d_12_238494!
batch_normalization_10_238497!
batch_normalization_10_238499!
batch_normalization_10_238501!
batch_normalization_10_238503
conv2d_13_238506
conv2d_13_238508!
batch_normalization_11_238511!
batch_normalization_11_238513!
batch_normalization_11_238515!
batch_normalization_11_238517
conv2d_14_238522
conv2d_14_238524!
batch_normalization_12_238527!
batch_normalization_12_238529!
batch_normalization_12_238531!
batch_normalization_12_238533
conv2d_15_238536
conv2d_15_238538!
batch_normalization_13_238541!
batch_normalization_13_238543!
batch_normalization_13_238545!
batch_normalization_13_238547
dense_4_238553
dense_4_238555
dense_5_238559
dense_5_238561
dense_6_238565
dense_6_238567
dense_7_238571
dense_7_238573
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_238441conv2d_8_238443*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2369772"
 conv2d_8/StatefulPartitionedCallĀ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_238446conv2d_9_238448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2370042"
 conv2d_9/StatefulPartitionedCallĆ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_7_238451batch_normalization_7_238453batch_normalization_7_238455batch_normalization_7_238457*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’..@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2370572/
-batch_normalization_7/StatefulPartitionedCall¤
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_2362902!
max_pooling2d_4/PartitionedCall
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_2371112
dropout_7/PartitionedCallĮ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_10_238462conv2d_10_238464*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2371352#
!conv2d_10/StatefulPartitionedCallÅ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_8_238467batch_normalization_8_238469batch_normalization_8_238471batch_normalization_8_238473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2371882/
-batch_normalization_8/StatefulPartitionedCallÕ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_11_238476conv2d_11_238478*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2372352#
!conv2d_11/StatefulPartitionedCallÅ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_9_238481batch_normalization_9_238483batch_normalization_9_238485batch_normalization_9_238487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2372882/
-batch_normalization_9/StatefulPartitionedCall„
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2365102!
max_pooling2d_5/PartitionedCall
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373422
dropout_8/PartitionedCallĮ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_12_238492conv2d_12_238494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2373662#
!conv2d_12/StatefulPartitionedCallĢ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_10_238497batch_normalization_10_238499batch_normalization_10_238501batch_normalization_10_238503*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23741920
.batch_normalization_10/StatefulPartitionedCallÖ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_13_238506conv2d_13_238508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2374662#
!conv2d_13/StatefulPartitionedCallĢ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_11_238511batch_normalization_11_238513batch_normalization_11_238515batch_normalization_11_238517*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23751920
.batch_normalization_11/StatefulPartitionedCall¦
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2367302!
max_pooling2d_6/PartitionedCall
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_2375732
dropout_9/PartitionedCallĮ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_14_238522conv2d_14_238524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2375972#
!conv2d_14/StatefulPartitionedCallĢ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_12_238527batch_normalization_12_238529batch_normalization_12_238531batch_normalization_12_238533*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_23765020
.batch_normalization_12/StatefulPartitionedCallÖ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_15_238536conv2d_15_238538*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2376972#
!conv2d_15/StatefulPartitionedCallĢ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_13_238541batch_normalization_13_238543batch_normalization_13_238545batch_normalization_13_238547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_23775020
.batch_normalization_13/StatefulPartitionedCall¦
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2369502!
max_pooling2d_7/PartitionedCall
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2378042
dropout_10/PartitionedCallų
flatten_1/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2378232
flatten_1/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_238553dense_4_238555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2378422!
dense_4/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2378752
dropout_11/PartitionedCall°
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_238559dense_5_238561*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2378992!
dense_5/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2379322
dropout_12/PartitionedCall°
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_6_238565dense_6_238567*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2379562!
dense_6/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_2379892
dropout_13/PartitionedCallÆ
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_7_238571dense_7_238573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2380132!
dense_7/StatefulPartitionedCall¾
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_238441*&
_output_shapes
:@*
dtype023
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp¾
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"conv2d_8/kernel/Regularizer/Square
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_8/kernel/Regularizer/Const¾
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/Sum
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!conv2d_8/kernel/Regularizer/mul/xĄ
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_8/kernel/Regularizer/mulö
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesī
ė:’’’’’’’’’00::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
³
«
C__inference_dense_7_layer_call_and_return_conditional_losses_240845

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_236290

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
»
F
*__inference_dropout_8_layer_call_fn_240036

inputs
identityĻ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_2373422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_7_layer_call_fn_236956

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2369502
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ą
serving_default¬
Q
conv2d_8_input?
 serving_default_conv2d_8_input:0’’’’’’’’’00;
dense_70
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:
Łå
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer-15
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
layer-29
layer_with_weights-18
layer-30
 	optimizer
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
Ł_default_save_signature
+Ś&call_and_return_all_conditional_losses
Ū__call__"ėÜ
_tf_keras_sequentialĖÜ{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 7.178980013122782e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¬

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+Ü&call_and_return_all_conditional_losses
Ż__call__"

_tf_keras_layerė	{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}}
ō	

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+Ž&call_and_return_all_conditional_losses
ß__call__"Ķ
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}
¼	
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+ą&call_and_return_all_conditional_losses
į__call__"ę
_tf_keras_layerĢ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}

;trainable_variables
<	variables
=regularization_losses
>	keras_api
+ā&call_and_return_all_conditional_losses
ć__call__"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ē
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+ä&call_and_return_all_conditional_losses
å__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
÷	

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+ę&call_and_return_all_conditional_losses
ē__call__"Š
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 64]}}
¾	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+č&call_and_return_all_conditional_losses
é__call__"č
_tf_keras_layerĪ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 128]}}
ł	

Rkernel
Sbias
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+ź&call_and_return_all_conditional_losses
ė__call__"Ņ
_tf_keras_layerø{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 128]}}
¾	
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]trainable_variables
^	variables
_regularization_losses
`	keras_api
+ģ&call_and_return_all_conditional_losses
ķ__call__"č
_tf_keras_layerĪ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 128]}}

atrainable_variables
b	variables
cregularization_losses
d	keras_api
+ī&call_and_return_all_conditional_losses
ļ__call__"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ē
etrainable_variables
f	variables
gregularization_losses
h	keras_api
+š&call_and_return_all_conditional_losses
ń__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ł	

ikernel
jbias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
+ņ&call_and_return_all_conditional_losses
ó__call__"Ņ
_tf_keras_layerø{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 128]}}
Ą	
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
+ō&call_and_return_all_conditional_losses
õ__call__"ź
_tf_keras_layerŠ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 256]}}
ł	

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+ö&call_and_return_all_conditional_losses
÷__call__"Ņ
_tf_keras_layerø{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 256]}}
Ē	
~axis
	gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
+ų&call_and_return_all_conditional_losses
ł__call__"ź
_tf_keras_layerŠ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 256]}}

trainable_variables
	variables
regularization_losses
	keras_api
+ś&call_and_return_all_conditional_losses
ū__call__"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ė
trainable_variables
	variables
regularization_losses
	keras_api
+ü&call_and_return_all_conditional_losses
ż__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ż	
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
+ž&call_and_return_all_conditional_losses
’__call__"Š
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 256]}}
Ē	
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"č
_tf_keras_layerĪ{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 512]}}
ż	
kernel
	bias
 trainable_variables
”	variables
¢regularization_losses
£	keras_api
+&call_and_return_all_conditional_losses
__call__"Š
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 512]}}
Ē	
	¤axis

„gamma
	¦beta
§moving_mean
Ømoving_variance
©trainable_variables
Ŗ	variables
«regularization_losses
¬	keras_api
+&call_and_return_all_conditional_losses
__call__"č
_tf_keras_layerĪ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 512]}}

­trainable_variables
®	variables
Æregularization_losses
°	keras_api
+&call_and_return_all_conditional_losses
__call__"š
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ķ
±trainable_variables
²	variables
³regularization_losses
“	keras_api
+&call_and_return_all_conditional_losses
__call__"Ų
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ģ
µtrainable_variables
¶	variables
·regularization_losses
ø	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ż
¹kernel
	ŗbias
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api
+&call_and_return_all_conditional_losses
__call__"Š
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
ķ
ætrainable_variables
Ą	variables
Įregularization_losses
Ā	keras_api
+&call_and_return_all_conditional_losses
__call__"Ų
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ū
Ćkernel
	Äbias
Åtrainable_variables
Ę	variables
Ēregularization_losses
Č	keras_api
+&call_and_return_all_conditional_losses
__call__"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ķ
Étrainable_variables
Ź	variables
Ėregularization_losses
Ģ	keras_api
+&call_and_return_all_conditional_losses
__call__"Ų
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ū
Ķkernel
	Ībias
Ļtrainable_variables
Š	variables
Ńregularization_losses
Ņ	keras_api
+&call_and_return_all_conditional_losses
__call__"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ķ
Ótrainable_variables
Ō	variables
Õregularization_losses
Ö	keras_api
+&call_and_return_all_conditional_losses
__call__"Ų
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ü
×kernel
	Ųbias
Łtrainable_variables
Ś	variables
Ūregularization_losses
Ü	keras_api
+&call_and_return_all_conditional_losses
__call__"Ļ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ņ
	Żiter
Žbeta_1
ßbeta_2

ądecay
įlearning_rate&m'm,m-m3m4mCmDmJmKmRmSmYmZmimjmpmqmxmym m”	m¢	m£	m¤	m„	m¦	m§	mØ	„m©	¦mŖ	¹m«	ŗm¬	Ćm­	Äm®	ĶmÆ	Īm°	×m±	Ųm²&v³'v“,vµ-v¶3v·4vøCv¹DvŗJv»Kv¼Rv½Sv¾YvæZvĄivĮjvĀpvĆqvÄxvÅyvĘvĒ	vČ	vÉ	vŹ	vĖ	vĢ	vĶ	vĪ	„vĻ	¦vŠ	¹vŃ	ŗvŅ	ĆvÓ	ÄvŌ	ĶvÕ	ĪvÖ	×v×	ŲvŲ"
	optimizer
×
&0
'1
,2
-3
34
45
C6
D7
J8
K9
R10
S11
Y12
Z13
i14
j15
p16
q17
x18
y19
20
21
22
23
24
25
26
27
„28
¦29
¹30
ŗ31
Ć32
Ä33
Ķ34
Ī35
×36
Ų37"
trackable_list_wrapper
Ķ
&0
'1
,2
-3
34
45
56
67
C8
D9
J10
K11
L12
M13
R14
S15
Y16
Z17
[18
\19
i20
j21
p22
q23
r24
s25
x26
y27
28
29
30
31
32
33
34
35
36
37
38
39
„40
¦41
§42
Ø43
¹44
ŗ45
Ć46
Ä47
Ķ48
Ī49
×50
Ų51"
trackable_list_wrapper
(
0"
trackable_list_wrapper
Ó
ānon_trainable_variables
ćlayer_metrics
!trainable_variables
"	variables
ämetrics
#regularization_losses
 ålayer_regularization_losses
ęlayers
Ū__call__
Ł_default_save_signature
+Ś&call_and_return_all_conditional_losses
'Ś"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
):'@2conv2d_8/kernel
:@2conv2d_8/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
µ
ēnon_trainable_variables
člayer_metrics
(trainable_variables
)	variables
émetrics
*regularization_losses
 źlayer_regularization_losses
ėlayers
Ż__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_9/kernel
:@2conv2d_9/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ģnon_trainable_variables
ķlayer_metrics
.trainable_variables
/	variables
īmetrics
0regularization_losses
 ļlayer_regularization_losses
šlayers
ß__call__
+Ž&call_and_return_all_conditional_losses
'Ž"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
.
30
41"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ńnon_trainable_variables
ņlayer_metrics
7trainable_variables
8	variables
ómetrics
9regularization_losses
 ōlayer_regularization_losses
õlayers
į__call__
+ą&call_and_return_all_conditional_losses
'ą"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
önon_trainable_variables
÷layer_metrics
;trainable_variables
<	variables
ųmetrics
=regularization_losses
 łlayer_regularization_losses
ślayers
ć__call__
+ā&call_and_return_all_conditional_losses
'ā"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ūnon_trainable_variables
ülayer_metrics
?trainable_variables
@	variables
żmetrics
Aregularization_losses
 žlayer_regularization_losses
’layers
å__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_10/kernel
:2conv2d_10/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Etrainable_variables
F	variables
metrics
Gregularization_losses
 layer_regularization_losses
layers
ē__call__
+ę&call_and_return_all_conditional_losses
'ę"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_8/gamma
):'2batch_normalization_8/beta
2:0 (2!batch_normalization_8/moving_mean
6:4 (2%batch_normalization_8/moving_variance
.
J0
K1"
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Ntrainable_variables
O	variables
metrics
Pregularization_losses
 layer_regularization_losses
layers
é__call__
+č&call_and_return_all_conditional_losses
'č"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_11/kernel
:2conv2d_11/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Ttrainable_variables
U	variables
metrics
Vregularization_losses
 layer_regularization_losses
layers
ė__call__
+ź&call_and_return_all_conditional_losses
'ź"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_9/gamma
):'2batch_normalization_9/beta
2:0 (2!batch_normalization_9/moving_mean
6:4 (2%batch_normalization_9/moving_variance
.
Y0
Z1"
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
]trainable_variables
^	variables
metrics
_regularization_losses
 layer_regularization_losses
layers
ķ__call__
+ģ&call_and_return_all_conditional_losses
'ģ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
atrainable_variables
b	variables
metrics
cregularization_losses
 layer_regularization_losses
layers
ļ__call__
+ī&call_and_return_all_conditional_losses
'ī"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
etrainable_variables
f	variables
metrics
gregularization_losses
 layer_regularization_losses
layers
ń__call__
+š&call_and_return_all_conditional_losses
'š"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_12/kernel
:2conv2d_12/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
ktrainable_variables
l	variables
 metrics
mregularization_losses
 ”layer_regularization_losses
¢layers
ó__call__
+ņ&call_and_return_all_conditional_losses
'ņ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_10/gamma
*:(2batch_normalization_10/beta
3:1 (2"batch_normalization_10/moving_mean
7:5 (2&batch_normalization_10/moving_variance
.
p0
q1"
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
£non_trainable_variables
¤layer_metrics
ttrainable_variables
u	variables
„metrics
vregularization_losses
 ¦layer_regularization_losses
§layers
õ__call__
+ō&call_and_return_all_conditional_losses
'ō"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_13/kernel
:2conv2d_13/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ønon_trainable_variables
©layer_metrics
ztrainable_variables
{	variables
Ŗmetrics
|regularization_losses
 «layer_regularization_losses
¬layers
÷__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_11/gamma
*:(2batch_normalization_11/beta
3:1 (2"batch_normalization_11/moving_mean
7:5 (2&batch_normalization_11/moving_variance
/
0
1"
trackable_list_wrapper
?
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
­non_trainable_variables
®layer_metrics
trainable_variables
	variables
Æmetrics
regularization_losses
 °layer_regularization_losses
±layers
ł__call__
+ų&call_and_return_all_conditional_losses
'ų"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
²non_trainable_variables
³layer_metrics
trainable_variables
	variables
“metrics
regularization_losses
 µlayer_regularization_losses
¶layers
ū__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
·non_trainable_variables
ølayer_metrics
trainable_variables
	variables
¹metrics
regularization_losses
 ŗlayer_regularization_losses
»layers
ż__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_14/kernel
:2conv2d_14/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
¼non_trainable_variables
½layer_metrics
trainable_variables
	variables
¾metrics
regularization_losses
 ælayer_regularization_losses
Ąlayers
’__call__
+ž&call_and_return_all_conditional_losses
'ž"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_12/gamma
*:(2batch_normalization_12/beta
3:1 (2"batch_normalization_12/moving_mean
7:5 (2&batch_normalization_12/moving_variance
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Įnon_trainable_variables
Ālayer_metrics
trainable_variables
	variables
Ćmetrics
regularization_losses
 Älayer_regularization_losses
Ålayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_15/kernel
:2conv2d_15/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ęnon_trainable_variables
Ēlayer_metrics
 trainable_variables
”	variables
Čmetrics
¢regularization_losses
 Élayer_regularization_losses
Źlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_13/gamma
*:(2batch_normalization_13/beta
3:1 (2"batch_normalization_13/moving_mean
7:5 (2&batch_normalization_13/moving_variance
0
„0
¦1"
trackable_list_wrapper
@
„0
¦1
§2
Ø3"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ėnon_trainable_variables
Ģlayer_metrics
©trainable_variables
Ŗ	variables
Ķmetrics
«regularization_losses
 Īlayer_regularization_losses
Ļlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Šnon_trainable_variables
Ńlayer_metrics
­trainable_variables
®	variables
Ņmetrics
Æregularization_losses
 Ólayer_regularization_losses
Ōlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Õnon_trainable_variables
Ölayer_metrics
±trainable_variables
²	variables
×metrics
³regularization_losses
 Ųlayer_regularization_losses
Łlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Śnon_trainable_variables
Ūlayer_metrics
µtrainable_variables
¶	variables
Ümetrics
·regularization_losses
 Żlayer_regularization_losses
Žlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_4/kernel
:2dense_4/bias
0
¹0
ŗ1"
trackable_list_wrapper
0
¹0
ŗ1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ßnon_trainable_variables
ąlayer_metrics
»trainable_variables
¼	variables
įmetrics
½regularization_losses
 ālayer_regularization_losses
ćlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
änon_trainable_variables
ålayer_metrics
ætrainable_variables
Ą	variables
ęmetrics
Įregularization_losses
 ēlayer_regularization_losses
člayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_5/kernel
:2dense_5/bias
0
Ć0
Ä1"
trackable_list_wrapper
0
Ć0
Ä1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
énon_trainable_variables
źlayer_metrics
Åtrainable_variables
Ę	variables
ėmetrics
Ēregularization_losses
 ģlayer_regularization_losses
ķlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
īnon_trainable_variables
ļlayer_metrics
Étrainable_variables
Ź	variables
šmetrics
Ėregularization_losses
 ńlayer_regularization_losses
ņlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_6/kernel
:2dense_6/bias
0
Ķ0
Ī1"
trackable_list_wrapper
0
Ķ0
Ī1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ónon_trainable_variables
ōlayer_metrics
Ļtrainable_variables
Š	variables
õmetrics
Ńregularization_losses
 ölayer_regularization_losses
÷layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ųnon_trainable_variables
łlayer_metrics
Ótrainable_variables
Ō	variables
śmetrics
Õregularization_losses
 ūlayer_regularization_losses
ülayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_7/kernel
:2dense_7/bias
0
×0
Ų1"
trackable_list_wrapper
0
×0
Ų1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
żnon_trainable_variables
žlayer_metrics
Łtrainable_variables
Ś	variables
’metrics
Ūregularization_losses
 layer_regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate

50
61
L2
M3
[4
\5
r6
s7
8
9
10
11
§12
Ø13"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
§0
Ø1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
æ

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"ø
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
.:,@2Adam/conv2d_8/kernel/m
 :@2Adam/conv2d_8/bias/m
.:,@@2Adam/conv2d_9/kernel/m
 :@2Adam/conv2d_9/bias/m
.:,@2"Adam/batch_normalization_7/gamma/m
-:+@2!Adam/batch_normalization_7/beta/m
0:.@2Adam/conv2d_10/kernel/m
": 2Adam/conv2d_10/bias/m
/:-2"Adam/batch_normalization_8/gamma/m
.:,2!Adam/batch_normalization_8/beta/m
1:/2Adam/conv2d_11/kernel/m
": 2Adam/conv2d_11/bias/m
/:-2"Adam/batch_normalization_9/gamma/m
.:,2!Adam/batch_normalization_9/beta/m
1:/2Adam/conv2d_12/kernel/m
": 2Adam/conv2d_12/bias/m
0:.2#Adam/batch_normalization_10/gamma/m
/:-2"Adam/batch_normalization_10/beta/m
1:/2Adam/conv2d_13/kernel/m
": 2Adam/conv2d_13/bias/m
0:.2#Adam/batch_normalization_11/gamma/m
/:-2"Adam/batch_normalization_11/beta/m
1:/2Adam/conv2d_14/kernel/m
": 2Adam/conv2d_14/bias/m
0:.2#Adam/batch_normalization_12/gamma/m
/:-2"Adam/batch_normalization_12/beta/m
1:/2Adam/conv2d_15/kernel/m
": 2Adam/conv2d_15/bias/m
0:.2#Adam/batch_normalization_13/gamma/m
/:-2"Adam/batch_normalization_13/beta/m
':%
2Adam/dense_4/kernel/m
 :2Adam/dense_4/bias/m
':%
2Adam/dense_5/kernel/m
 :2Adam/dense_5/bias/m
':%
2Adam/dense_6/kernel/m
 :2Adam/dense_6/bias/m
&:$	2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
.:,@2Adam/conv2d_8/kernel/v
 :@2Adam/conv2d_8/bias/v
.:,@@2Adam/conv2d_9/kernel/v
 :@2Adam/conv2d_9/bias/v
.:,@2"Adam/batch_normalization_7/gamma/v
-:+@2!Adam/batch_normalization_7/beta/v
0:.@2Adam/conv2d_10/kernel/v
": 2Adam/conv2d_10/bias/v
/:-2"Adam/batch_normalization_8/gamma/v
.:,2!Adam/batch_normalization_8/beta/v
1:/2Adam/conv2d_11/kernel/v
": 2Adam/conv2d_11/bias/v
/:-2"Adam/batch_normalization_9/gamma/v
.:,2!Adam/batch_normalization_9/beta/v
1:/2Adam/conv2d_12/kernel/v
": 2Adam/conv2d_12/bias/v
0:.2#Adam/batch_normalization_10/gamma/v
/:-2"Adam/batch_normalization_10/beta/v
1:/2Adam/conv2d_13/kernel/v
": 2Adam/conv2d_13/bias/v
0:.2#Adam/batch_normalization_11/gamma/v
/:-2"Adam/batch_normalization_11/beta/v
1:/2Adam/conv2d_14/kernel/v
": 2Adam/conv2d_14/bias/v
0:.2#Adam/batch_normalization_12/gamma/v
/:-2"Adam/batch_normalization_12/beta/v
1:/2Adam/conv2d_15/kernel/v
": 2Adam/conv2d_15/bias/v
0:.2#Adam/batch_normalization_13/gamma/v
/:-2"Adam/batch_normalization_13/beta/v
':%
2Adam/dense_4/kernel/v
 :2Adam/dense_4/bias/v
':%
2Adam/dense_5/kernel/v
 :2Adam/dense_5/bias/v
':%
2Adam/dense_6/kernel/v
 :2Adam/dense_6/bias/v
&:$	2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
ī2ė
!__inference__wrapped_model_236180Å
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *5¢2
0-
conv2d_8_input’’’’’’’’’00
ī2ė
H__inference_sequential_1_layer_call_and_return_conditional_losses_239288
H__inference_sequential_1_layer_call_and_return_conditional_losses_239083
H__inference_sequential_1_layer_call_and_return_conditional_losses_238181
H__inference_sequential_1_layer_call_and_return_conditional_losses_238036Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
-__inference_sequential_1_layer_call_fn_238436
-__inference_sequential_1_layer_call_fn_239506
-__inference_sequential_1_layer_call_fn_238690
-__inference_sequential_1_layer_call_fn_239397Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ī2ė
D__inference_conv2d_8_layer_call_and_return_conditional_losses_239529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_conv2d_8_layer_call_fn_239538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_conv2d_9_layer_call_and_return_conditional_losses_239549¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_conv2d_9_layer_call_fn_239558¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239596
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239660
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239578
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239642“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
6__inference_batch_normalization_7_layer_call_fn_239622
6__inference_batch_normalization_7_layer_call_fn_239609
6__inference_batch_normalization_7_layer_call_fn_239686
6__inference_batch_normalization_7_layer_call_fn_239673“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
³2°
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_236290ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
0__inference_max_pooling2d_4_layer_call_fn_236296ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Č2Å
E__inference_dropout_7_layer_call_and_return_conditional_losses_239698
E__inference_dropout_7_layer_call_and_return_conditional_losses_239703“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
*__inference_dropout_7_layer_call_fn_239713
*__inference_dropout_7_layer_call_fn_239708“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_10_layer_call_and_return_conditional_losses_239724¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_10_layer_call_fn_239733¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239753
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239835
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239817
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239771“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
6__inference_batch_normalization_8_layer_call_fn_239784
6__inference_batch_normalization_8_layer_call_fn_239861
6__inference_batch_normalization_8_layer_call_fn_239797
6__inference_batch_normalization_8_layer_call_fn_239848“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_11_layer_call_and_return_conditional_losses_239872¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_11_layer_call_fn_239881¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239919
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239983
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239965
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239901“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
6__inference_batch_normalization_9_layer_call_fn_240009
6__inference_batch_normalization_9_layer_call_fn_239932
6__inference_batch_normalization_9_layer_call_fn_239996
6__inference_batch_normalization_9_layer_call_fn_239945“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
³2°
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_236510ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
0__inference_max_pooling2d_5_layer_call_fn_236516ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Č2Å
E__inference_dropout_8_layer_call_and_return_conditional_losses_240026
E__inference_dropout_8_layer_call_and_return_conditional_losses_240021“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
*__inference_dropout_8_layer_call_fn_240031
*__inference_dropout_8_layer_call_fn_240036“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_12_layer_call_and_return_conditional_losses_240047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_12_layer_call_fn_240056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240158
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240140
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240076
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240094“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
7__inference_batch_normalization_10_layer_call_fn_240107
7__inference_batch_normalization_10_layer_call_fn_240184
7__inference_batch_normalization_10_layer_call_fn_240171
7__inference_batch_normalization_10_layer_call_fn_240120“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_13_layer_call_and_return_conditional_losses_240195¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_13_layer_call_fn_240204¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240224
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240288
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240242
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240306“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
7__inference_batch_normalization_11_layer_call_fn_240332
7__inference_batch_normalization_11_layer_call_fn_240319
7__inference_batch_normalization_11_layer_call_fn_240255
7__inference_batch_normalization_11_layer_call_fn_240268“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
³2°
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_236730ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
0__inference_max_pooling2d_6_layer_call_fn_236736ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Č2Å
E__inference_dropout_9_layer_call_and_return_conditional_losses_240344
E__inference_dropout_9_layer_call_and_return_conditional_losses_240349“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
*__inference_dropout_9_layer_call_fn_240359
*__inference_dropout_9_layer_call_fn_240354“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_14_layer_call_and_return_conditional_losses_240370¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_14_layer_call_fn_240379¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240399
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240417
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240481
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240463“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
7__inference_batch_normalization_12_layer_call_fn_240507
7__inference_batch_normalization_12_layer_call_fn_240494
7__inference_batch_normalization_12_layer_call_fn_240443
7__inference_batch_normalization_12_layer_call_fn_240430“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_15_layer_call_and_return_conditional_losses_240518¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_15_layer_call_fn_240527¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240611
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240565
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240629
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240547“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
7__inference_batch_normalization_13_layer_call_fn_240642
7__inference_batch_normalization_13_layer_call_fn_240655
7__inference_batch_normalization_13_layer_call_fn_240591
7__inference_batch_normalization_13_layer_call_fn_240578“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
³2°
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_236950ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
0__inference_max_pooling2d_7_layer_call_fn_236956ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ź2Ē
F__inference_dropout_10_layer_call_and_return_conditional_losses_240672
F__inference_dropout_10_layer_call_and_return_conditional_losses_240667“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_10_layer_call_fn_240682
+__inference_dropout_10_layer_call_fn_240677“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
E__inference_flatten_1_layer_call_and_return_conditional_losses_240688¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_flatten_1_layer_call_fn_240693¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_dense_4_layer_call_and_return_conditional_losses_240704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_4_layer_call_fn_240713¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_11_layer_call_and_return_conditional_losses_240730
F__inference_dropout_11_layer_call_and_return_conditional_losses_240725“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_11_layer_call_fn_240740
+__inference_dropout_11_layer_call_fn_240735“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ķ2ź
C__inference_dense_5_layer_call_and_return_conditional_losses_240751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_5_layer_call_fn_240760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_12_layer_call_and_return_conditional_losses_240777
F__inference_dropout_12_layer_call_and_return_conditional_losses_240772“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_12_layer_call_fn_240787
+__inference_dropout_12_layer_call_fn_240782“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ķ2ź
C__inference_dense_6_layer_call_and_return_conditional_losses_240798¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_6_layer_call_fn_240807¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_13_layer_call_and_return_conditional_losses_240824
F__inference_dropout_13_layer_call_and_return_conditional_losses_240819“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_13_layer_call_fn_240829
+__inference_dropout_13_layer_call_fn_240834“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ķ2ź
C__inference_dense_7_layer_call_and_return_conditional_losses_240845¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_7_layer_call_fn_240854¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
³2°
__inference_loss_fn_0_240865
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *¢ 
:B8
$__inference_signature_wrapper_238815conv2d_8_inputē
!__inference__wrapped_model_236180ĮK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×Ų?¢<
5¢2
0-
conv2d_8_input’’’’’’’’’00
Ŗ "1Ŗ.
,
dense_7!
dense_7’’’’’’’’’Ź
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240076tpqrs<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ź
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240094tpqrs<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 ļ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240140pqrsN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ļ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_240158pqrsN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ¢
7__inference_batch_normalization_10_layer_call_fn_240107gpqrs<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’¢
7__inference_batch_normalization_10_layer_call_fn_240120gpqrs<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’Ē
7__inference_batch_normalization_10_layer_call_fn_240171pqrsN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ē
7__inference_batch_normalization_10_layer_call_fn_240184pqrsN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’ņ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240224N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ņ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240242N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ķ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240288w<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ķ
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_240306w<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ź
7__inference_batch_normalization_11_layer_call_fn_240255N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ź
7__inference_batch_normalization_11_layer_call_fn_240268N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’„
7__inference_batch_normalization_11_layer_call_fn_240319j<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’„
7__inference_batch_normalization_11_layer_call_fn_240332j<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’ó
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240399N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240417N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ī
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240463x<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ī
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240481x<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ė
7__inference_batch_normalization_12_layer_call_fn_240430N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_12_layer_call_fn_240443N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’¦
7__inference_batch_normalization_12_layer_call_fn_240494k<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’¦
7__inference_batch_normalization_12_layer_call_fn_240507k<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’ó
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240547„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240565„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ī
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240611x„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ī
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_240629x„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ė
7__inference_batch_normalization_13_layer_call_fn_240578„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_13_layer_call_fn_240591„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’¦
7__inference_batch_normalization_13_layer_call_fn_240642k„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’¦
7__inference_batch_normalization_13_layer_call_fn_240655k„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’Ē
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239578r3456;¢8
1¢.
(%
inputs’’’’’’’’’..@
p
Ŗ "-¢*
# 
0’’’’’’’’’..@
 Ē
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239596r3456;¢8
1¢.
(%
inputs’’’’’’’’’..@
p 
Ŗ "-¢*
# 
0’’’’’’’’’..@
 ģ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2396423456M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ģ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2396603456M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
6__inference_batch_normalization_7_layer_call_fn_239609e3456;¢8
1¢.
(%
inputs’’’’’’’’’..@
p
Ŗ " ’’’’’’’’’..@
6__inference_batch_normalization_7_layer_call_fn_239622e3456;¢8
1¢.
(%
inputs’’’’’’’’’..@
p 
Ŗ " ’’’’’’’’’..@Ä
6__inference_batch_normalization_7_layer_call_fn_2396733456M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ä
6__inference_batch_normalization_7_layer_call_fn_2396863456M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@É
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239753tJKLM<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 É
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239771tJKLM<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 ī
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239817JKLMN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ī
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239835JKLMN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ”
6__inference_batch_normalization_8_layer_call_fn_239784gJKLM<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’”
6__inference_batch_normalization_8_layer_call_fn_239797gJKLM<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’Ę
6__inference_batch_normalization_8_layer_call_fn_239848JKLMN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ę
6__inference_batch_normalization_8_layer_call_fn_239861JKLMN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239901YZ[\N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ī
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239919YZ[\N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 É
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239965tYZ[\<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 É
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239983tYZ[\<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ę
6__inference_batch_normalization_9_layer_call_fn_239932YZ[\N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ę
6__inference_batch_normalization_9_layer_call_fn_239945YZ[\N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’”
6__inference_batch_normalization_9_layer_call_fn_239996gYZ[\<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’”
6__inference_batch_normalization_9_layer_call_fn_240009gYZ[\<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’¶
E__inference_conv2d_10_layer_call_and_return_conditional_losses_239724mCD7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_10_layer_call_fn_239733`CD7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ "!’’’’’’’’’·
E__inference_conv2d_11_layer_call_and_return_conditional_losses_239872nRS8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_11_layer_call_fn_239881aRS8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’·
E__inference_conv2d_12_layer_call_and_return_conditional_losses_240047nij8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_12_layer_call_fn_240056aij8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’·
E__inference_conv2d_13_layer_call_and_return_conditional_losses_240195nxy8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_13_layer_call_fn_240204axy8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’¹
E__inference_conv2d_14_layer_call_and_return_conditional_losses_240370p8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_14_layer_call_fn_240379c8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’¹
E__inference_conv2d_15_layer_call_and_return_conditional_losses_240518p8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_conv2d_15_layer_call_fn_240527c8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’“
D__inference_conv2d_8_layer_call_and_return_conditional_losses_239529l&'7¢4
-¢*
(%
inputs’’’’’’’’’00
Ŗ "-¢*
# 
0’’’’’’’’’..@
 
)__inference_conv2d_8_layer_call_fn_239538_&'7¢4
-¢*
(%
inputs’’’’’’’’’00
Ŗ " ’’’’’’’’’..@“
D__inference_conv2d_9_layer_call_and_return_conditional_losses_239549l,-7¢4
-¢*
(%
inputs’’’’’’’’’..@
Ŗ "-¢*
# 
0’’’’’’’’’..@
 
)__inference_conv2d_9_layer_call_fn_239558_,-7¢4
-¢*
(%
inputs’’’’’’’’’..@
Ŗ " ’’’’’’’’’..@§
C__inference_dense_4_layer_call_and_return_conditional_losses_240704`¹ŗ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
(__inference_dense_4_layer_call_fn_240713S¹ŗ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’§
C__inference_dense_5_layer_call_and_return_conditional_losses_240751`ĆÄ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
(__inference_dense_5_layer_call_fn_240760SĆÄ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’§
C__inference_dense_6_layer_call_and_return_conditional_losses_240798`ĶĪ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
(__inference_dense_6_layer_call_fn_240807SĶĪ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¦
C__inference_dense_7_layer_call_and_return_conditional_losses_240845_×Ų0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
(__inference_dense_7_layer_call_fn_240854R×Ų0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ø
F__inference_dropout_10_layer_call_and_return_conditional_losses_240667n<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 ø
F__inference_dropout_10_layer_call_and_return_conditional_losses_240672n<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 
+__inference_dropout_10_layer_call_fn_240677a<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’
+__inference_dropout_10_layer_call_fn_240682a<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’Ø
F__inference_dropout_11_layer_call_and_return_conditional_losses_240725^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 Ø
F__inference_dropout_11_layer_call_and_return_conditional_losses_240730^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 
+__inference_dropout_11_layer_call_fn_240735Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
+__inference_dropout_11_layer_call_fn_240740Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’Ø
F__inference_dropout_12_layer_call_and_return_conditional_losses_240772^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 Ø
F__inference_dropout_12_layer_call_and_return_conditional_losses_240777^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 
+__inference_dropout_12_layer_call_fn_240782Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
+__inference_dropout_12_layer_call_fn_240787Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’Ø
F__inference_dropout_13_layer_call_and_return_conditional_losses_240819^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 Ø
F__inference_dropout_13_layer_call_and_return_conditional_losses_240824^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 
+__inference_dropout_13_layer_call_fn_240829Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
+__inference_dropout_13_layer_call_fn_240834Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’µ
E__inference_dropout_7_layer_call_and_return_conditional_losses_239698l;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ "-¢*
# 
0’’’’’’’’’@
 µ
E__inference_dropout_7_layer_call_and_return_conditional_losses_239703l;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ "-¢*
# 
0’’’’’’’’’@
 
*__inference_dropout_7_layer_call_fn_239708_;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ " ’’’’’’’’’@
*__inference_dropout_7_layer_call_fn_239713_;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ " ’’’’’’’’’@·
E__inference_dropout_8_layer_call_and_return_conditional_losses_240021n<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 ·
E__inference_dropout_8_layer_call_and_return_conditional_losses_240026n<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_dropout_8_layer_call_fn_240031a<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’
*__inference_dropout_8_layer_call_fn_240036a<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’·
E__inference_dropout_9_layer_call_and_return_conditional_losses_240344n<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 ·
E__inference_dropout_9_layer_call_and_return_conditional_losses_240349n<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 
*__inference_dropout_9_layer_call_fn_240354a<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’
*__inference_dropout_9_layer_call_fn_240359a<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’«
E__inference_flatten_1_layer_call_and_return_conditional_losses_240688b8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
*__inference_flatten_1_layer_call_fn_240693U8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "’’’’’’’’’;
__inference_loss_fn_0_240865&¢

¢ 
Ŗ " ī
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_236290R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_4_layer_call_fn_236296R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_236510R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_5_layer_call_fn_236516R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_236730R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_6_layer_call_fn_236736R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_236950R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_7_layer_call_fn_236956R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
H__inference_sequential_1_layer_call_and_return_conditional_losses_238036½K&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×ŲG¢D
=¢:
0-
conv2d_8_input’’’’’’’’’00
p

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_1_layer_call_and_return_conditional_losses_238181½K&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×ŲG¢D
=¢:
0-
conv2d_8_input’’’’’’’’’00
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_1_layer_call_and_return_conditional_losses_239083µK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×Ų?¢<
5¢2
(%
inputs’’’’’’’’’00
p

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_1_layer_call_and_return_conditional_losses_239288µK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×Ų?¢<
5¢2
(%
inputs’’’’’’’’’00
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ā
-__inference_sequential_1_layer_call_fn_238436°K&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×ŲG¢D
=¢:
0-
conv2d_8_input’’’’’’’’’00
p

 
Ŗ "’’’’’’’’’ā
-__inference_sequential_1_layer_call_fn_238690°K&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×ŲG¢D
=¢:
0-
conv2d_8_input’’’’’’’’’00
p 

 
Ŗ "’’’’’’’’’Ś
-__inference_sequential_1_layer_call_fn_239397ØK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×Ų?¢<
5¢2
(%
inputs’’’’’’’’’00
p

 
Ŗ "’’’’’’’’’Ś
-__inference_sequential_1_layer_call_fn_239506ØK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×Ų?¢<
5¢2
(%
inputs’’’’’’’’’00
p 

 
Ŗ "’’’’’’’’’ü
$__inference_signature_wrapper_238815ÓK&',-3456CDJKLMRSYZ[\ijpqrsxy„¦§Ø¹ŗĆÄĶĪ×ŲQ¢N
¢ 
GŖD
B
conv2d_8_input0-
conv2d_8_input’’’’’’’’’00"1Ŗ.
,
dense_7!
dense_7’’’’’’’’’