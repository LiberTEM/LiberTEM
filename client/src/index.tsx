/* eslint-disable no-underscore-dangle */
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import { applyMiddleware, compose, createStore } from 'redux';
import createSagaMiddleware from 'redux-saga';
import App from './App';
import { rootSaga } from './sagas';
import { rootReducer } from './store';

const sagaMiddleware = createSagaMiddleware();

declare global {
    interface Window { __REDUX_DEVTOOLS_EXTENSION_COMPOSE__: typeof compose }
}

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

const store = createStore(rootReducer, composeEnhancers(
    applyMiddleware(
        sagaMiddleware,
    )
));

const container = document.getElementById('root') as HTMLElement;
const root = createRoot(container);

root.render(
    <Provider store={store}>
        <App />
    </Provider>,
);

sagaMiddleware.run(rootSaga);
