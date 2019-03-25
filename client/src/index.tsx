// tslint:disable:ordered-imports
import "core-js/fn/symbol/iterator.js";
import "core-js/es6/symbol.js";
import 'react-app-polyfill/ie11';
// tslint:enable:ordered-imports

import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { applyMiddleware, compose, createStore } from 'redux';
import createSagaMiddleware from 'redux-saga';
import App from './App';
import registerServiceWorker from './registerServiceWorker';
import { rootSaga } from './sagas';
import { rootReducer } from './store';

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