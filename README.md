Python 3.6+
pandas
seaborn
numpy
matplotlib
itertools


Task 01: Reasoning

>>> cd reasoning
>>> python run.py <value of M>  <value of N>


To plot the results:

>>> python showplot.py


Task 02: Learning

>>> cd learning
>>> python run.py <dataset> <learning rate> <alpha> <beta> <percentage of rules used>

where <dataset> is among:
 - "muta"
 - "krk"
 - "uwcse_i1"
 - "alzAmi"
 - "alzSco"
 - "alzCho"
 - "alzTox"
 - "alzAmi"

<learning rate> is the value for the learning rate

<alpha> is the balance weight for the generative cost

<beta> is the balance weight for the discriminative cost

<percentage of rule used> identifies how many rules will used (e.g. 0.5 means 50% of the rules will be used)


Task 03: SAT

>>> cd sat
>>> python run.py --dnf "<path to dimacs file>"

for example: python run.py --dnf "./sr5/grp1/sr_n=0010_pk2=0.30_pg=0.40_t=3_sat=1.dimacs"

