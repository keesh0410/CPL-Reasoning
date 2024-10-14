FEWSHOT_INST = r"""You are a powerful agent with advanced reasoning and planning capabilities. Answer the questions as best you can.

!!!Remember:
1. Your answer should have two sections: "Plans" and "Detailed Implementation".
2. In the "Plans" section, you should outline step-by-step plans for solving the problem. These plans might include extracting key information, forming sub-questions, analyzing aspects, etc. Each step should introduce new insights, avoid overly abstract or generic actions. End each step with "<endstep>".
3. In the "Detailed Implementation" section, provide detailed steps that correspond to each plan, and conclude with "The final answer is \boxed{answer}.<endsolution>"

The following is a template for your answer:

Question: The input question

Plans:
Plan 1: Describe the first plan step.<endstep>
Plan 2: Describe the second plan step<endstep>
...
Plan N: Describe the final plan step<endstep>

Detailed Implementation:
1. Execute the first plan step
2. Execute the second plan step
...
N. Execute the final plan step
The final answer is \boxed{answer}.<endsolution>

The following are 2 demonstration examples.

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Plans:
Plan 1: Analyze the total number of clips sold in April.<endstep>
Plan 2: Calculate the number of clips sold in May by applying the "half as many" condition to the number sold in April.<endstep>
Plan 3: Sum the results from April and May to determine the overall total of clips sold over the two months.<endstep>

Detailed Implementation:
1. Natalia sold 48 clips in April. 
2. The number of clips sold in May is $\frac{48}{2}=24$.
3. The total number of clips sold in April and May combined is $48+24=72$.
The final answer is \boxed{72}.<endsolution>

Question: If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?

Plans:
Plan 1: Understand that the equation $x^2+y^2=1$ defines a circle centered at the origin with a radius of 1. To maximize $|x|+|y|$, we need to consider points on this circle that maximize the sum of the absolute values of $x$ and $y$.<endstep>
Plan 2: Recognize that $|x|+|y|$ is maximized when both $|x|$ and $|y|$ are large. The maximum sum occurs along lines where $x$ and $y$ contribute equally, specifically along the lines $y=x$ and $y=-x$.<endstep>
Plan 3: Identify the points of intersection between the lines $y=x$ and $y=-x$ with the circle $x^2+y^2=1$. These points are expected to yield the maximum value of $|x|+|y|$.<endstep>
Plan 4: Evaluate $|x|+|y|$ for the intersection points to determine the maximum possible value.<endstep>

Detailed Implementation:
1. The circle $x^2+y^2=1$ is centered at the origin with a radius of 1. We need to find the points on this circle that maximize the sum $|x|+|y|$.
2. To maximize $|x|+|y|$, the sum is largest when both $|x|$ and $|y|$ are large. This occurs along the lines $y=x$ and $y=-x$, where $x$ and $y$ contribute equally to the sum.
3. The intersection points are $\left(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, $\left(\frac{1}{\sqrt{2}},-\frac{1}{\sqrt{2}}\right)$, $\left(-\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, and $\left(-\frac{1}{\sqrt{2}},-\frac{1}{\sqrt{2}}\right)$.
4. For these points, calculate $|x|+|y|$. For $\left(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, $|x|+|y|=\sqrt{2}$. The same value applies to the other points. Therefore, the maximum value is $\sqrt{2}$.
The final answer is $\boxed{\sqrt{2}}$.<endsolution>

Now! It's your turn.

"""


FEWSHOT_XML = r"""<question>
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
</question>
<plan>
<step>
Plan 1: Analyze the total number of clips sold in April.
</step>
<step>
Plan 2: Calculate the number of clips sold in May by applying the "half as many" condition to the number sold in April.
</step>
<step>
Plan 3: Sum the results from April and May to determine the overall total of clips sold over the two months.
</step>
</plan>
<solution>
1. Natalia sold 48 clips in April.
2. The number of clips sold in May is $\frac{48}{2}=24$.
3. The total number of clips sold in April and May combined is $48+24=72$.
The final answer is \boxed{72}.
</solution>

<question>
If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?
</question>
<plan>
<step>
Plan 1: Understand that the equation $x^2+y^2=1$ defines a circle centered at the origin with a radius of 1. To maximize $|x|+|y|$, we need to consider points on this circle that maximize the sum of the absolute values of $x$ and $y.
</step>
<step>
Plan 2: Recognize that $|x|+|y|$ is maximized when both $|x|$ and $|y|$ are large. The maximum sum occurs along lines where $x$ and $y$ contribute equally, specifically along the lines $y=x$ and $y=-x.
</step>
<step>
Plan 3: Identify the points of intersection between the lines $y=x$ and $y=-x$ with the circle $x^2+y^2=1$. These points are expected to yield the maximum value of $|x|+|y|.
</step>
<step>
Plan 4: Evaluate $|x|+|y|$ for the intersection points to determine the maximum possible value.
</step>
</plan>
<solution>
1. The circle $x^2+y^2=1$ is centered at the origin with a radius of 1. We need to find the points on this circle that maximize the sum $|x|+|y|$.
2. To maximize $|x|+|y|$, the sum is largest when both $|x|$ and $|y|$ are large. This occurs along the lines $y=x$ and $y=-x$, where $x$ and $y$ contribute equally to the sum.
3. The intersection points are $\left(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, $\left(\frac{1}{\sqrt{2}},-\frac{1}{\sqrt{2}}\right)$, $\left(-\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, and $\left(-\frac{1}{\sqrt{2}},-\frac{1}{\sqrt{2}}\right)$.
4. For these points, calculate $|x|+|y|$. For $\left(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$, $|x|+|y|=\sqrt{2}$. The same value applies to the other points. Therefore, the maximum value is $\sqrt{2}$.
The final answer is $\boxed{\sqrt{2}}.
</solution>

"""