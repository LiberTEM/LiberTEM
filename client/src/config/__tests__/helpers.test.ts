import { test, expect, describe } from 'vitest';

import { joinPaths, makeUnique } from '../helpers';
import { initialConfigState } from '../reducers';

describe('joinPaths', () => {
    test('properly joins paths', () => {
        expect(joinPaths(initialConfigState, '/', 'home')).toBe("/home");
        expect(joinPaths(initialConfigState, '/home', 'something')).toBe("/home/something");
        expect(joinPaths(initialConfigState, '/home', '..')).toBe("/home/..");
    });
});

describe('makeUnique', () => {
    test('keep the order', () => {
        expect(makeUnique([1, 1, 2, 7, 4, 5, 1, 9, 3])).toEqual([1, 2, 7, 4, 5, 9, 3]);
    });
});