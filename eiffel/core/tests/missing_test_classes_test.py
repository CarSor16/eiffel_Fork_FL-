"""A predicted class absent from test must not disappear from the confusion matrix."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd
from eiffel.core.client import EiffelClient
from eiffel.datasets.preprocessed_network import PreprocessedNetworkDataset


def test_confusion_matrix_accounts_for_predictions_of_absent_test_class(monkeypatch):
    dataset = PreprocessedNetworkDataset(
        X=pd.DataFrame({'x':[0.,1.,2.]}),
        y=pd.Series([0,1,1], dtype='int64'),
        m=pd.DataFrame({'ClassName':['A','B','B'], 'ClassId':[0,1,1]}),
        key='missing-test-class', _default_target=['*'],
    )
    model=Mock()
    model.metrics_names=['loss','accuracy']
    model.evaluate.return_value=[0.5, 1/3]
    model.predict.return_value=np.array([[0.,0.,1.],[0.,1.,0.],[0.,0.,1.]])
    client=object.__new__(EiffelClient)
    client.model=model;client.cid='benign:0';client.seed=2026;client.verbose=0
    client.data_holder=SimpleNamespace(get=SimpleNamespace(remote=lambda split:dataset))
    monkeypatch.setattr('eiffel.core.client.ray.get',lambda obj:obj)
    _,n,payload=client.evaluate([], {'batch_size':2, 'capture_inference':False})
    cm=np.array(json.loads(payload['confusion_matrix']))
    assert cm.sum()==n==3
    assert cm.shape==(3,3)
    assert cm[0,2]==1 and cm[1,2]==1
    assert json.loads(payload['confusion_matrix_labels'])==[0,1,2]
    global_metrics=json.loads(payload['global'])
    assert global_metrics['num_classes']==3
    assert global_metrics['num_observed_test_classes']==2
