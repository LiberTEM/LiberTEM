import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import App from './App';
import { rootSaga } from './sagas';
import { store, sagaMiddleware } from './store';

const container = document.getElementById('root') as HTMLElement;
const root = createRoot(container);

root.render(
    <Provider store={store}>
        <App />
    </Provider>,
);

sagaMiddleware.run(rootSaga);
