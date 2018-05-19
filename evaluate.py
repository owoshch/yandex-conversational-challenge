import pandas as pd
from separate_ranking_model import RankingModel
from model.config import Config
from model.data_utils import load_pairwise_testset, rank_predictions, save_submission


config = Config()

model = RankingModel(config)
model.build()
model.restore_session(config.dir_model)

sub = load_pairwise_testset(model.config.path_to_preprocessed_private)
sub_df = pd.read_csv(model.config.path_to_preprocessed_private)

print ('Path to submission:', model.config.path_to_submission)

sub_preds = model.predict_proba(sub, sub_df)

save_submission(model.config.path_to_submission, sub_df, rank_predictions(sub_df, sub_preds))
