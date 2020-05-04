import { Actions } from "../actions";
import { configReducer, initialConfigState } from "../reducers";

const somePath = "/some/path/"

describe('browser reducer', () => {
    it('should allow to add a favorite', () => {
        const newState = configReducer(initialConfigState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual(["/some/path/"]);
    });
    it('should allow to remove a favorite', () => {
        const intermediateState = configReducer(initialConfigState, Actions.toggleStar(somePath));
        const newState = configReducer(intermediateState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual([]);
    });
})
