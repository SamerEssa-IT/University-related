# LegalNER
Named entity recognition for legal documents in German 

# LSTM model

## SequenceEncoder:
  New methode is implemented:
    
  #SequenceEncoder.reverse(X): 
  decodes the encoded data 
    input(X) : type list or DataFrame.series
    return:   the original form of the data
        

    def reverse(self, X):
        if isinstance(X,pd.Series):
            return X.apply(self._reverse)
        if type(X) == list:
            return [self._reverse(sequence) for sequence in X]

    def _reverse(self, sequence):
        processed_list = [self.idx_to_word.get(token,0) for token in sequence ]
        processed_list = [token for token in processed_list if token != self.special_tokens[0]]
        return processed_list



## Training court dataset with LSTM

#### Input Data:

Court sentences and labels tokinized and embedded using SequenceEncoder

The resulting representation of court Sentences :

          >0        [2740, 11, 2817, 18, 11822, 3894, 11, 792, 0, ...
          >1        [50, 238, 20, 750, 17, 2740, 1329, 3016, 17, 2...
          >2        [404, 50, 238, 20, 750, 17, 2740, 19, 395, 188...
          >3        [29, 2370, 240, 20, 2817, 122, 3, 678, 6, 598,...
          >                                ...                        
          >10202    [320, 8, 29, 6872, 58, 2909, 25, 15420, 1718, ...
          >10203    [144, 1432, 168, 6, 304, 9, 51, 11, 35, 57, 11...
          >10204    [5987, 21, 2, 101, 34, 43, 39, 7341, 1697, 39,...
          >10205    [12983, 21, 5, 7139, 58, 3460, 193, 2, 72, 53,...

The prediction results of the model returns binary representation of the labels,
therefore, the each encoded labels was transformed into a unique binary vectors    
The resulting representation of labels :       

   >array[[0., 1., 0., ..., 0., 0., 0.],
   >     [0., 1., 0., ..., 0., 0., 0.],
   >     [0., 1., 0., ..., 0., 0., 0.],
   >                  ...,
   >     [1., 0., 0., ..., 0., 0., 0.],
   >     [1., 0., 0., ..., 0., 0., 0.],
   >     [1., 0., 0., ..., 0., 0., 0.]]
   >     
   >array[[0., 1., 0., ..., 0., 0., 0.],
   >     [0., 1., 0., ..., 0., 0., 0.],
   >     [0., 1., 0., ..., 0., 0., 0.],
   >                 ...,
   >     [1., 0., 0., ..., 0., 0., 0.],
   >     [1., 0., 0., ..., 0., 0., 0.],
   >     [1., 0., 0., ..., 0., 0., 0.]]
        
            


## Model: "model"
>_________________________________________________________________
>Layer (type)                 Output Shape              Param # 
>
>=================================================================
>input_1 (InputLayer)         [(None, 100)]             0         
>_________________________________________________________________
>embedding (Embedding)        (None, 100, 100)          8158300   
>_________________________________________________________________
>spatial_dropout1d (SpatialDr (None, 100, 100)          0         
>_________________________________________________________________
>bidirectional (Bidirectional (None, 100, 200)          160800    
>_________________________________________________________________
>time_distributed (TimeDistri (None, 100, 40)           8040     
>
>=================================================================
>Total params: 8,327,140
>Trainable params: 8,327,140
>Non-trainable params: 0
>_________________________________________________________________
        

## Cross Validation:
A self implemented cross validation that calls the model F time (f is the number of foldings) each time with diffrent distribution of train and test data/labels
and returns the best split of the data, aswell as the model it was fitted on and it's history
        
## resulting Dataframe of the predcition:

![SCREEN~1](https://user-images.githubusercontent.com/74178017/108269458-4c935f80-716e-11eb-878f-2e119304aee9.PNG)


## Evaluation of the prediction 

![resukt_lstm](https://user-images.githubusercontent.com/74178017/110324430-36aee700-8016-11eb-839c-a5241aeed9be.png)

