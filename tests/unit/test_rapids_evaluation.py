import cudf as cu
import numpy as np

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)
from reco_utils.evaluation.rapids_evaluation import (
    ramse,
    mae,
)

TOL = 0.0001


@pytest.fixture
def rating_true():
    return cu.DataFrame(
        [
            (DEFAULT_USER_COL, [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            (DEFAULT_ITEM_COL, [
                1,
                2,
                3,
                1,
                4,
                5,
                6,
                7,
                2,
                5,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
            ]),
            (DEFAULT_RATING_COL, [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1]),
        ]
    )


@pytest.fixture
def rating_pred():
    return cu.DataFrame(
        [
            (DEFAULT_USER_COL, [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            (DEFAULT_ITEM_COL, [
                3,
                10,
                12,
                10,
                3,
                5,
                11,
                13,
                4,
                10,
                7,
                13,
                1,
                3,
                5,
                2,
                11,
                14,
            ]),
            (DEFAULT_PREDICTION_COL, [
                14,
                13,
                12,
                14,
                13,
                12,
                11,
                10,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
            ]),
            (DEFAULT_RATING_COL, [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1]),
        ]
    )


def test_rapids_merge_rating(rating_true, rating_pred):
    y_true, y_pred = merge_rating_true_pred(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    )
#   To convert into np.array, y_true.to_array()
    target_y_true = np.array([3, 3, 5, 5, 3, 3, 2, 1])
    target_y_pred = np.array([14, 12, 7, 8, 13, 6, 11, 5])

    assert y_true.shape == y_pred.shape
    assert np.all(y_true.to_array() == target_y_true)
    assert np.all(y_pred.to_array() == target_y_pred)

    
def test_rapids_rmse(rating_true, rating_pred):
    assert (
        rmse(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
        )
        == 0
    )
    assert rmse(rating_true, rating_pred) == pytest.approx(7.254309, TOL)

    
def test_rapids_mae(rating_true, rating_pred):
    assert (
        mae(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
        )
        == 0
    )
    assert mae(rating_true, rating_pred) == pytest.approx(6.375, TOL)

    
def test_rapids_merge_ranking(rating_true, rating_pred):
    data_hit, data_hit_count, n_users = merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        relevancy_method="top_k",
    )

    assert isinstance(data_hit, cu.DataFrame)

    assert isinstance(data_hit_count, cu.DataFrame)
    columns = data_hit_count.columns
    columns_exp = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL]
    assert set(columns).intersection(set(columns_exp)) is not None

    assert n_users == 3