import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { applyMiddleware, combineReducers, compose, createStore } from 'redux';
import createSagaMiddleware from 'redux-saga';
import App from './App';
import { channelStatusReducer } from './channel/reducers';
import { datasetReducer } from './dataset/reducers';
import { jobReducer } from './job/reducers';
import registerServiceWorker from './registerServiceWorker';
import { rootSaga } from './sagas';


const rootReducer = combineReducers({
    channelStatus: channelStatusReducer,
    dataset: datasetReducer,
    job: jobReducer,
})
const sagaMiddleware = createSagaMiddleware();

const composeEnhancers = (window as any).__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

const store = createStore(rootReducer, composeEnhancers(
    applyMiddleware(
        sagaMiddleware,
    )
));

ReactDOM.render(
    <Provider store={store}>
        <App />
    </Provider>,
  document.getElementById('root') as HTMLElement
);
registerServiceWorker();

sagaMiddleware.run(rootSaga);