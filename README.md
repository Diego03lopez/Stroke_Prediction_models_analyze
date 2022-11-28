# Stroke Prediction models analyze
![](https://github.com/Diego03lopez/Stroke_Prediction_models_analyze/blob/main/Stroke.png)

# Introduction
The problem that is expected to be solved is to be able to generate a supervised learning model of classification that allows predicting the risk that a person has in his daily life of suffering a stroke from the analysis of different personal and health characteristics of the same as it could be verified previously, and thus be able to take measures to improve some of the described parameters that classify it among that group of person at risk, something that allows to make people aware of the danger in which they can be found because of this, where a priority is given to the health of the person with respect to some changes in his personal life to avoid being in the risk group known as "stroke".

# Dataset
To solve this problem from different models it is necessary to evaluate their predictive performance from evaluation metrics by means of a specific dataset found [here](https://www.kaggle.com/code/jorgeromn/brain-stroke-with-random-forest-accuracy-97/data?select=full_data.csv "here").

#### Understanding dataset
Among its characteristics, such as the data that will allow classifying the risk or not of a person who may suffer a stroke, is found:
- Gender: It is divided between male and female.
- Age: Corresponds to the age of each of the subjects from whom this variety of data was taken.
- Hypertension: this characteristic is classified in a binary way, since it is described as a "1" if the person suffers from hypertension, and "0" if the person       does not suffer in any way from this condition.
- Heart disease: Like hypertension, it is referred to in the same way by giving positive for any heart disease with "1" and otherwise "0".
- Ever married: This is nothing more than a string type data that affirms with "Yes" if the person has been married or "No" otherwise.
- Type of work: This characteristic could not be missing since it can be understood that according to a specific job, one can have more or less possibilities of         suffering from this condition, since with more exhausting and stressful jobs one can suffer from hypertension and lead directly to a position closer to being           classified as at risk for a stroke; this is classified by 3 types which are Private, Government or Independent.
- Type of residence between Urban and Rural
- Average glucose level: In general: Less than 100 mg/dL (5.6 mmol/L) is considered normal. Between 100 and 125 mg/dL (5.6 to 6.9 mmol/L) is diagnosed as                 prediabetes.
- BMI: This measurement indicates that it helps us to know if we have a correct weight with respect to our height, therefore, this number is based on both weight         and height, which for adults over 20 years of age can be classified as follows.

- Smoking status: corresponds to a group of 4 possibilities which are Unknown, Former smoker, Smoked and Non-smoker.
- Stroke (Outcome): Indicates the possibility of suffering a stroke by "1" as positive, and "0" negative not possible, which will be the column taken as labels to       classify each of the rows of data in the risk described above.

`$ npm install marked`
