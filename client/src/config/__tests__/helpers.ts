import { joinPaths, makeUnique } from '../helpers';
import { initialConfigState } from '../reducers';

describe('joinPaths', () => {
    it('properly joins paths', () => {
        expect(joinPaths(initialConfigState, '/', 'home')).toBe("/home");
        expect(joinPaths(initialConfigState, '/home', 'something')).toBe("/home/something");
        expect(joinPaths(initialConfigState, '/home', '..')).toBe("/home/..");
    });
});

describe('makeUnique', () => {
    it('keep the order', () => {
        expect(makeUnique([1, 1, 2, 7, 4, 5, 1, 9, 3])).toEqual([1, 2, 7, 4, 5, 9, 3]);
    });
});