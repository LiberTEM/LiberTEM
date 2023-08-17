import { render } from '@testing-library/react';
import { test, describe } from 'vitest'
import { Provider } from 'react-redux';
import { store } from './store';
import App from './App';

describe('App', () => {
  test('renders without crashing', () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );
    // screen.debug();
  });
})
