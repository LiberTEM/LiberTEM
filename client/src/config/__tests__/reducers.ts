import { ClusterTypes } from "../../messages";
import { Actions } from "../actions";
import { configReducer, ConfigState } from "../reducers";

const somePath = "/some/path/"

const initialState: ConfigState = {
    separator: '/', version: '', revision: '', cwd: '/',
    lastOpened: {},
    fileHistory: [],
    resultFileFormats: {},
    localCores: 0,
    haveConfig: true,
    lastConnection: {
        type: ClusterTypes.LOCAL,
        address: "tcp://localhost:8786",
    },
    starred: [],
};

describe('browser reducer', () => {
    it('should allow to add a favorite', () => {
        const newState = configReducer(initialState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual(["/some/path/"]);
    });
    it('should allow to remove a favorite', () => {
        const intermediateState = configReducer(initialState, Actions.toggleStar(somePath));
        const newState = configReducer(intermediateState, Actions.toggleStar(somePath));
        expect(newState.starred).toEqual([]);
    });
})
