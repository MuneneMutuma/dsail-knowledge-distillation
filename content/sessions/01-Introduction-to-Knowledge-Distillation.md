---
title: "Introduction to Knowledge Distillation"
---

# Fundamentals
## What is Knowledge Distillation?


>  Knowledge distillation is a machine learning technique that aims to transfer the learnings of a large pre-trained model, the “teacher model,” to a smaller “student model.”
>  
> *[IBM](https://www.ibm.com/think/topics/knowledge-distillation#:~:text=Knowledge%20distillation%20is%20a%20machine%20learning%20technique%20that%20aims%20to%20transfer%20the%20learnings%20of%20a%20large%20pre%2Dtrained%20model%2C%20the%20%E2%80%9Cteacher%20model%2C%E2%80%9D%20to%20a%20smaller%20%E2%80%9Cstudent%20model.%E2%80%9D)* 

> In machine learning, knowledge distillation or model distillation is the process of transferring knowledge from a large model to a smaller one.
> 
> *[Wikipedia](https://en.wikipedia.org/wiki/Knowledge_distillation#:~:text=In%20machine%20learning%2C%20knowledge%20distillation%20or%20model%20distillation%20is%20the%20process%20of%20transferring%20knowledge%20from%20a%20large%20model%20to%20a%20smaller%20one.)* 

> Knowledge distillation is a technique that enables knowledge transfer from large, computationally expensive models to smaller ones without losing validity. This allows for deployment on less powerful hardware, making evaluation faster and more efficient.
> 
> [*pytorch*](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#:~:text=Knowledge%20distillation%20is%20a%20technique%20that%20enables%20knowledge%20transfer%20from%20large%2C%20computationally%20expensive%20models%20to%20smaller%20ones%20without%20losing%20validity.%20This%20allows%20for%20deployment%20on%20less%20powerful%20hardware%2C%20making%20evaluation%20faster%20and%20more%20efficient.)


So what are the components of knowledge distillation?
1. knowledge
2. larger model - teacher
3. smaller model - student

But the big question is why?

### Why Knowledge Distillation?
The aim of KD is to get a smaller model to match the output of a larger model with the aim of saving on computational cost while retaining high performance relative to the larger model. This is especially important for less powerful hardware.

### So, what is knowledge?
KD is anchored on the concept of transfering *knowledge* from teacher to student. The question then is what is this *knowledge*? Let us first look at normal deep learning training under *cross entropy loss*. We have a model and training data. The training objective, i.e. the loss, is defined below as:

$$
L_{CE} = - \sum_{i=1}^K y_i\log(p_i)
$$

where $y_i$ is the true label and is typically one-hot encoded and $p_i$ is the model's predicted probability for class $i$. Thus, for a one-hot target, this reduces for a single example to:
$$
L_{CE}= -\log(p_y)
$$
where $p_y$ is the predicted probability assigned to the correct class.

Notice then that under cross entropy loss training, the model receives a signal only about the correct class. The loss contains no explicit information about how the other classes relate to each other or to the correct one.

What knowledge distillation proposes is the idea that, a model already trained on the data, has captured some interclass relationship and information that is not directly inferrable from the ground truth, but it has learned through training and is captured in its state and reflected in its outputs. The model thus does not just output a 1 for the correct class and 0 for all other classes, but a probability distribution over all classes which is a reflection of how the classes relate to each other relative to the specific output. It essentially tells us something about how the classes relate to each other that the ground truth alone could not. This is the concept of dark knowledge. So then how do we formulate this as a loss for the student?

$$
L_{KD} = - T^2\sum_{i=1}^Kp_i^T(T)\log{p_i^S(T)}
$$

where:
$$
p_i^T(T)=\frac{\exp(\frac{z_i^T}{T})}{\sum_{j=1}^K\exp(\frac{z_j^T}{T})}
$$

and 
$$
p_i^S(T)=\frac{\exp(\frac{z_i^S}{T})}{\sum_{j=1}^K\exp(\frac{z_j^S}{T})}
$$

Notice that this is cross entropy still, but now against the teacher distribution and not the ground truth.

So what is this "knowledge"?


![[Pasted image 20260816184410.jpg|700]]

  Say we have an image classification task, and the horse picture above is part of our dataset. The ground truth is that this is 100% a horse and nothing else. So the ground truth can be represented as a one hot encoding of this where the horse has a probability of 1 and the rest 0.

![[Pasted image 20260816184029.png|700]]


Now, let's train our student on this and see what the student says:

![[Pasted image 20260816184127.png]]

"*it's a dog*", which is wrong, the student is still learning. So let's do the loss calculation:

![[Pasted image 20260816184209.png|700]]


Notice that because all the ground truth probabilities for all other classes is zero, they contribute nothing to the loss, and the student is made only that this is a horse and that's it. This is clearly evident from the formula:

$$
L_{CE} = - \sum_{i=1}^K \mathbf{y_i}\log(p_i)
$$
notice $y_i$ pre-multiplies every log probability in this summation. So all the places where the ground truth is zero becomes zero and contributes nothing to the oss.

But now let's take a look at the teacher. The teacher outputs a probability distribution saying: "I think this is a horse with 55% certainty, but there's a 22% probability it may be a zebra, a 14% probability it may be a cow, 6% a dog, and 3% an airplane". 

![[Pasted image 20260816184050.png|700]]

This output gives the highest probability to the correct class, which is okay, but it tells us more: *it tells us also how these classes relate to each other based on their probabilities for each class for a given output*. The teacher has, in the process of training, learnt some latent interclass relationships present in the data that are not explicitly in the ground truth. Therefore, if we train the student on this, the student will have the advantage of the experience of the teacher and not only the ground truth.

So how does this loss, conceptually or intuitively look like?

![[Pasted image 20260816184220.png|700]]

$$
L_{KD} = -T^2 \sum_{i=1}^Kp_i^T(T)\log{p_i^S(T)}
$$

Ideally we calculate the log of the probabilities of the student and weight them with the probabilities of the teacher for each class. Notice then that most of the time, the value $p_i^T(T)$ is a value greater than zero, and thus will contribute something to the loss. In this case for example, the student will learn that, "okay, it's not dog, it's a horse, but a horse is similar to a zebra in certain respects". Essentially, each loss calculation based on this gives the student more information compared to with the ground truth only.

### T for Temperature
Now, there is an important dial for all this, a hyperparameter called **temperature**. For us to understand temperature, we will go back to information theory briefly, we will look at entropy. Remember entropy encapsulate the amount of *uncertainty* for a given event occurring, and consequently it encapsulates the amount of *information* that is gained by the occurrence of a given event. It can also be thought of as the amount of *surprise* on the occurrence of an event. So *higher uncertainty* means *higher information* and also *higher surprise*. The more uncertain an event is the more surprising is its occurrence and the more information we gain by its occurrence.

Now, let's go back to our case scenario first to see this.

![[Pasted image 20260816184029.png|700]]

Here the entropy is very low because the uncertainty is very low, it can only be a horse, so we do not learn as much information in this case, it's not as surprising that it is a horse, it is very deterministic.

But look at this:
![[Pasted image 20260816184050.png|700]]

This gives us more information, the uncertainty is higher, and hence the amount of information is more. Comparing this with the previous, the student will learn more from this, than from the previous one.


*But*, at times the teacher can be very confident of its answer, thus it will be more or less like the ground truth, and will hence reduce the amount of information that the student gets. If for example the teacher is 95% sure that this is a horse, then it is more or less roughly equal to the ground truth and there is not much for the student to learn.

This is where *temperature* comes in. Temperature softens the teacher's probability distribution, generally increasing its entropy and making the relative probabilities of non-target classes more informative to the student.

So let's look at those two examples. Remember the formula for entropy:

$$
H(p) =-\sum_ip_i\log(p_i)
$$

so if our teacher is very confident and has a distribution like:

![[Pasted image 20260816192949.png]]

$$

\begin{aligned}
H(\mathbf{p})
&=
-\left[
0.95\log_2(0.95)
+
0.03\log_2(0.03)
+
0.009\log_2(0.009)
+
0.005\log_2(0.005)
+
0.006\log_2(0.006)
\right]\\[4pt]
&=
-\left[
-0.07030
-0.15177
-0.06116
-0.03822
-0.04428
\right]\\[4pt]
&\approx 0.36573
\end{aligned}
$$


Now these are the softmax probabilities gotten as:
$$
p_i =\frac{e^{z_i}}{\sum_je^{z_j}}
$$

When we apply temperature, it becomes:

$$
p_i(T) =\frac{e^{z_i/T}}{\sum_je^{z_j/T}}
$$

For this illustration, take $z_i=\ln(p_i)$, which reproduces the original distribution at T=1

$$
\begin{aligned}
z_1 &= \ln(0.95) \approx -0.05129\\
z_2 &= \ln(0.03) \approx -3.50656\\
z_3 &= \ln(0.009) \approx -4.71053\\
z_4 &= \ln(0.005) \approx -5.29832\\
z_5 &= \ln(0.006) \approx -5.115996
\end{aligned}
$$

Then apply $T=2$ on the softmax:

$$
\begin{aligned}
e^{z_1/2} &= e^{-0.05129/2} \approx 0.97467\\
e^{z_2/2} &= e^{-3.50656/2} \approx 0.17321\\
e^{z_3/2} &= e^{-4.71053/2} \approx 0.09487\\
e^{z_4/2} &= e^{-5.29832/2} \approx 0.07071\\
e^{z_5/2} &= e^{-5.115996/2} \approx 0.07746
\end{aligned}
$$

$$
0.97467+0.17321+0.09487+0.07071+0.07746≈1.39092.
$$
and now we apply softmax on the temperature scaled figures:

$$
\begin{aligned}
p_1(2)&=\frac{0.97467}{1.39092}\approx0.70074\\
p_2(2)&=\frac{0.17321}{1.39092}\approx0.12453\\
p_3(2)&=\frac{0.09487}{1.39092}\approx0.06821\\
p_4(2)&=\frac{0.07071}{1.39092}\approx0.05084\\
p_5(2)&=\frac{0.07746}{1.39092}\approx0.05569
\end{aligned}
$$

Thus:
$$
p(T=2)=[0.70,0.12,0.07,0.05,0.06]
$$
and the entropy after temperature scaling:

$$
\begin{aligned}
H(\mathbf p_{T=2})
&=
-\Big[
0.70074\log_2(0.70074)
+0.12453\log_2(0.12453)\\
&\qquad
+0.06821\log_2(0.06821)
+0.05084\log_2(0.05084)\\
&\qquad
+0.05569\log_2(0.05569)
\Big]\\
&\approx 1.44852\text{ bits}.
\end{aligned}
$$


contrast that with
$$
p(T=1) =[0.95,0.03,0.009,0.005,0.006]
$$
$$
\begin{aligned}
H(\mathbf p)
&=
-\Big[
0.95\log_2(0.95)
+0.03\log_2(0.03)
+0.009\log_2(0.009)\\
&\qquad\qquad
+0.005\log_2(0.005)
+0.006\log_2(0.006)
\Big]\\
&\approx 0.36573\text{ bits}.
\end{aligned}
$$

That is the effect and purpose of temperature.

One last point on this is that *the loss formula is scaled by $T^2$ because $T$ leads to smaller gradients proportional to $\frac{1}{T^2}$*.


### Full Loss Formula
Having understood the foundations of knowledge distillation and the losses, we have to put them together now. $L_{KD}$ is not usually used alone in the loss for distillation, rather it is a combination of $L_{CE}$ and $L_{KD}$ as follows:

$$
\text{loss} = \alpha L_{CE} + (1-\alpha)L_{KD}
$$

where alpha is a hyperparameter just like $T$. Thus, the full loss is a weighted sum of the cross entropy against ground truth, and $L_{KD}$.


### Types of Distillation based on Knowledge
There are different types of knowledge that can be distilled, and we have primarily looked at one: *logit- or response-based distillation*. This is distillation off of the raw logits of a teacher model. However, there are others: *feature-based* and *relation-based*

*Feature-based* transfers knowledge from the teacher's intermediate layers/representations instead of only on the final output. The student is trained to approximate as close as possible some of the internal feature representation of the teacher:

$$
f_T(x)\approx f_S(x)
$$
where $f_T(x)$ and $f_S(x)$ are corresponding teacher and student feature representations for a select layer. Sometimes the teacher and student may have different architectures or feature dimensions necessitating the need of transformations before comparison and distillation.

*Relation-based* on the other hand aims to get the student to approximate the relations of the teacher on inputs. That is, given a set of inputs, instead of focusing on individual data points in isolation, it transfers structural relationships between them, as observed in the teacher. Typically, relation functions used are distance and angle.

### Paradigms of Distillation
There are also several methods of doing distillation based on how the teacher is dealt with. There is *online*, *offline*, and *self* distillation. In *offline* distillation, the teacher is pre-trained and frozen, and the student learns from the pre-computed outputs of the teacher. In *online* distillation, both the teacher and the student are being trained at the same time. On the other hand, for *self distillation* does not require a separate teacher but rather the distillation signal comes from another generation, stage, branch or prediction of the same model. 

### Eval
Now, once distillation is accomplished, we evaluate the student model on different dimensions. We look at the predictive performance, teacher-student fidelity,  efficiency/compression, latency, calibration and any other metric relevant to the field of application.

Predictive performance may be measured by metrics such as accuracy, precision, recall, F1 scores, perplexity, e.t.c. Teacher-student fidelity measures how closely the student reproduces the teacher. It looks at the probability distribution outputs of the students against those of the teacher and measures how close the student matches them. Compression is measured as a ratio either of parameters or raw storage size, but the former is more often used. For edge deployment, where speed and real time response is critical, latency is critical and is a key metric in evaluating the student model. Calibration is also key in assessing the model so that it is not overconfident. These are some of the few but many others may be added. It is good to have a multifaceted assessment for a fuller, richer more informative view.

Finally, there is ablation. This is where a student trained from scratch without knowledge distillation is also compared against the student trained via knowledge distillation to assess the effectiveness of distillation itself.