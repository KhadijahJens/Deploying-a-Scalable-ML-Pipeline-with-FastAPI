# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was specifically tailored for the census.csv dataset, aiming to build a classification model. Employing a random forest classifier, the model was trained with default configurations.

## Intended Use

To utilize this model effectively, it's designed for predicting "workclass" based on "education". You can execute it via the command line by running "python train_model.py".

## Training Data

The data used for training this model is sourced from publicly available Census Bureau data. Specifically, 80% of this data is allocated for training purposes.

## Evaluation Data

The data used in this model originates from publicly available Census Bureau data. A portion comprising 20% of this data is reserved for model validation. Categorical features and the target label underwent transformation using the One Hot Encoder and label binarizer, respectively. These transformations were applied based on fittings from the training set.

## Metrics

The model was evaluated using Precision, Recall, and FBeta. The overall performance on these metrics was: Precision: 0.7419, Recall: 0.6384, FBeta: 0.6863 Model saved to model/model.pkl. Model saved to model/encoder.pkl

## Ethical Considerations

The data might contain biases influenced by various factors such as demographics and unaccounted variables during data collection. It also uses data slices which may skew results. As a result, it may not fully represent the entire population.

## Caveats and Recommendations

Addressing potential data imbalances and ensuring the data reflects current economic conditions, such as adjusting for inflation, are crucial steps to mitigate feature bias and improve model accuracy. Updating the data regularly ensures its relevance and reliability for predictive tasks.