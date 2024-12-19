# Neural Reasoning for Sure through Constructing Explainable Models

## The supplementary document is ```sup_doc.pdf```;
## Paremeters are set in ```params.json```;

# Step 1. Download sample output dataset ```hsphnn_runs.zip``` 

[a sample experiment outputs can be downloaded at https://figshare.com/articles/dataset/Experiment_outputs_of_HSphNN/28060331](https://figshare.com/articles/dataset/Experiment_outputs_of_HSphNN/28060331)

# Step 2. Unzip ```hsphnn_runs.zip``` 

type following command to unzip ```hsphnn_runs.zip``` and move the data directory 
```
$ unzip hsphnn_runs.zip
```
After that, the directory structure is as follows.
```
--
 |-/data/hsphnn_runs/
 |-/HSphNN/ 
      |--/config/
      |--/data/
      |--/Syllogism
      |--/ValidSyllogism
```
# Step 3. to see the results of the first experiment (HSphNN achieves the symbolic-level of syllogistic reasoning with 1 epoch), type

```
$  python eval_exp1.py
```

The results are saved in the following files.
```
|--data
     |--hsphnn_runs
            |--ValidSyllogism_DIM2R1InitLoc0
            |--ValidSyllogism_DIM3R1InitLoc0
            |--ValidSyllogism_DIM15R1InitLoc0
            |--ValidSyllogism_DIM30R1InitLoc0 
            |--ValidSyllogism_DIM100R1InitLoc0
            |--ValidSyllogism_DIM200R1InitLoc0
            |--ValidSyllogism_DIM1000R1InitLoc0
            |--ValidSyllogism_DIM2000R1InitLoc0   
            |--ValidSyllogism_DIM3000R1InitLoc0     
```
```DIM3000``` means that spheres have the 3000 dimensions.
```R1``` means that spheres are initialised with their centres being located at the surface of a sphere with the radius of ```1```.
```InitLoc0``` means that spheres are initialised as being co-incided.

## Experiment result  

When spheres are initialised being coincided, HSphNN successfully identified all 24 valid syllogistic reasoning in one epoch. The dimension of sphere ranges from 2 to 3000.
```
+---------------------+-------+-----------------------------+--------------+
| Dimension of HSphere | epoch | #Identified Valid Reasoning | Total Number |
+---------------------+-------+-----------------------------+--------------+
|          2          |   1   |              24             |      24      |
|          3          |   1   |              24             |      24      |
|          15         |   1   |              24             |      24      |
|          30         |   1   |              24             |      24      |
|         100         |   1   |              24             |      24      |
|         200         |   1   |              24             |      24      |
|         1000        |   1   |              24             |      24      |
|         2000        |   1   |              24             |      24      |
|         3000        |   1   |              24             |      24      |
+---------------------+-------+-----------------------------+--------------+
```

Experiments show that it took HSphNN more time to determine a valid syllogistic reasoning than to determine an invalid one.

```
+-------------------+---------------------+----------------------------+---------+--------+----------+
| type of syllogism | number of syllogism | max time cost (in seconds) |  min -  | mean - | median - |
+-------------------+---------------------+----------------------------+---------+--------+----------+
|  valid syllogism  |         240         |           67.55            |   2.81  | 17.71  |  12.09   |
| invalid syllogism |         2320        |           66.29            | 0.00069 |  3.46  |   2.41   |
|   all syllogism   |         2560        |           67.55            | 0.00069 |  4.80  |   2.52   |
+-------------------+---------------------+----------------------------+---------+--------+----------+
```

 HSphNN is scalable with the increase of the dimension.

![alt text](pic/time4valid.png)

![alt text](pic/time4invalid.png)

![alt text](pic/time4all.png)

# Step 4. to see the results of the second experiment (HSphNN gave feedback to ChatGPT)

## create a ```config``` directory in the ```HSphNN``` directory, and create the ```openai_key.txt``` file to save your API key of [openai](https://platform.openai.com/docs/quickstart). 
## Then type
```
$  python eval_exp2.py
```
## Experiment results
```
****************************************************************************************************
using GPT-3.5-turbo, maximum 2 time feedback
../data/hsphnn_runs/ChatGPT3tb_short2_SphNN_words/
```
```_words``` at the end of the directory name means that syllogistic statements use meaningful words, such as ```all Greeks are human```.

The following table is the performance of ChatGPT gpt-3.5-turbo without the feedback from HSphNN.
```
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      103      |    58    |    41    |    11    |       15       |   28  |       0.40234375       |      0.48828125      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
```
The following table is the performance of ChatGPT gpt-3.5-turbo with maximum two round of the feedback from HSphNN.

```
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      128      |    49    |    18    |    6     |       6        |   49  |          0.5           |      0.30859375      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
```
The following table lists the number of correct responses of ChatGPT gpt-3.5-turbo at the i-th feedback.

```
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+----+---+
|               #Num of feedback               |  0  | 1  | 2 |
+----------------------------------------------+-----+----+---+
| #Tasks with correct decision and explanation | 103 | 17 | 8 |
+----------------------------------------------+-----+----+---+
```


```
../data/hsphnn_runs/ChatGPT3tb_short2_SphNN_symbol/
```
```_symbol``` at the end of the directory name means that syllogistic statements use simple symbols, such as ```all S are M0```.
```
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      120      |    0     |    73    |    33    |       14       |   16  |        0.46875         |       0.46875        |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      151      |    2     |    30    |    19    |       4        |   50  |       0.58984375       |      0.21484375      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+----+----+
|               #Num of feedback               |  0  | 1  | 2  |
+----------------------------------------------+-----+----+----+
| #Tasks with correct decision and explanation | 120 | 17 | 14 |
+----------------------------------------------+-----+----+----+
```


```
../data/hsphnn_runs/ChatGPT3tb_short2_SphNN_random/
```
```_random``` at the end of the directory name means that syllogistic statements use random symbols, such as ```All HWsF1eq9 are hONvNxop```.

```
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      106      |    1     |    45    |    38    |       17       |   49  |       0.4140625        |      0.39453125      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      138      |    4     |    20    |    19    |       1        |   74  |       0.5390625        |       0.171875       |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+----+----+
|               #Num of feedback               |  0  | 1  | 2  |
+----------------------------------------------+-----+----+----+
| #Tasks with correct decision and explanation | 106 | 20 | 12 |
+----------------------------------------------+-----+----+----+
```

```
****************************************************************************************************
using GPT-4o, maximum 2 time feedback
```
The following tables were the performances of ChatGPT-4o. 

```
../data/hsphnn_runs/ChatGPT4o_short2_SphNN_words/
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      191      |    15    |    8     |    20    |       17       |   5   |       0.74609375       |       0.234375       |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      194      |    20    |    10    |    10    |       17       |   5   |       0.7578125        |      0.22265625      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+---+---+
|               #Num of feedback               |  0  | 1 | 2 |
+----------------------------------------------+-----+---+---+
| #Tasks with correct decision and explanation | 191 | 3 | 0 |
+----------------------------------------------+-----+---+---+
../data/hsphnn_runs/ChatGPT4o_short2_SphNN_symbol/
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      196      |    3     |    13    |    20    |       19       |   5   |        0.765625        |      0.21484375      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      202      |    8     |    13    |    9     |       23       |   1   |       0.7890625        |      0.20703125      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+---+---+
|               #Num of feedback               |  0  | 1 | 2 |
+----------------------------------------------+-----+---+---+
| #Tasks with correct decision and explanation | 196 | 5 | 1 |
+----------------------------------------------+-----+---+---+
../data/hsphnn_runs/ChatGPT4o_short2_SphNN_random/
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      211      |    1     |    4     |    16    |       18       |   6   |       0.82421875       |      0.15234375      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 2 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      217      |    4     |    2     |    9     |       18       |   6   |       0.84765625       |      0.12890625      |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+---+---+
|               #Num of feedback               |  0  | 1 | 2 |
+----------------------------------------------+-----+---+---+
| #Tasks with correct decision and explanation | 211 | 4 | 2 |
+----------------------------------------------+-----+---+---+
****************************************************************************************************
```

```
using GPT-4o, maximum 10 time feedback
../data/hsphnn_runs/ChatGPT4o_short2_SphNN_random10F/
```
We increase the maximum number of feedback to 10 round and found that ChatGPT was not sensitive to the repetition. 
```
no feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      211      |    0     |    6     |    17    |       17       |   5   |       0.82421875       |       0.15625        |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
with maximum 10 times feedback
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
| #Correct+Expl | #HALLU 0 | #HALLU 1 | #HALLU 2 | #Inorrect+Expl | #Rest | #Ratio of Correct+Expl | #Irrationality ratio |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
|      218      |    6     |    6     |    4     |       18       |   4   |       0.8515625        |      0.1328125       |
+---------------+----------+----------+----------+----------------+-------+------------------------+----------------------+
Number of correct decision and correct explanation vs. the number of feedbacks
+----------------------------------------------+-----+---+---+---+---+---+---+---+---+---+----+
|               #Num of feedback               |  0  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
+----------------------------------------------+-----+---+---+---+---+---+---+---+---+---+----+
| #Tasks with correct decision and explanation | 211 | 5 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0  |
+----------------------------------------------+-----+---+---+---+---+---+---+---+---+---+----+
```

