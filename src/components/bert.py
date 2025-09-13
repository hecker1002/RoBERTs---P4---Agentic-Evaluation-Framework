''' fine tune the BERT on 
{ prompt ,response  , is_bias , is_toxic ,is_following_rule_regex}-> label( hallucinate score , info accurate score )
trying to the prompt ,resposne pair with basic info (from regex match) -> actual semantic meanign based  score 
 '''


from transformers import BertForSequenceClassification, BertTokenizer,Trainer 

tokenizer = BertTokenizer.from_pretrained( 'bert-base-uncased' )
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased'   , num_labels = 2 ) 
# 2( for halauciant + info accuracy score )

# fine tune the bert here
trainer  = Trainer( training_args = training_args , model  = bert_model , dataset = dataset  , epochs  = 10 )

trainer.train()