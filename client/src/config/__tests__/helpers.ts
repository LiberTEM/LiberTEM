import { joinPaths, makeUnique } from '../helpers'
import { ConfigState } from '../reducers';

const config: ConfigState = {
    separator: '/', version: '', revision: '', cwd: '/',
    lastOpened: {},
    fileHistory: [],
    resultFileFormats: {},
    localCores: 0,
    haveConfig: true,
};

describe('joinPaths', () => {
    it('properly joins paths', () => {
        expect(joinPaths(config, '/', 'home')).toBe("/home");
        expect(joinPaths(config, '/home', 'something')).toBe("/home/something");
        expect(joinPaths(config, '/home', '..')).toBe("/home/..");
    });
});

describe('makeUnique', () => {
    it('keep the order', () => {
        expect(makeUnique([1, 1, 2, 7, 4, 5, 1, 9, 3])).toEqual([1, 2, 7, 4, 5, 9, 3]);
    });
});