import { test, expect, describe } from 'vitest';

import { Actions } from "../actions";
import { configReducer, initialConfigState } from "../reducers";

const somePath = "/some/path/"

describe('browser reducer', () => {
    test('should allow to add a favorite', () => {
        const newState = configReducer(initialConfigState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual(["/some/path/"]);
    });
    test('should allow to remove a favorite', () => {
        const intermediateState = configReducer(initialConfigState, Actions.toggleStar(somePath));
        const newState = configReducer(intermediateState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual([]);
    });
})
