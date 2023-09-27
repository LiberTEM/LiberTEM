import { describe, expect, it } from "vitest";
import { parseHashParameters, splitLikePython } from "../helpers";

describe('splitLikePython', () => {
    it('supports exact matches', () => {
        expect(
            splitLikePython("a=b", "=", 1)
        ).toEqual(["a", "b"]);
    })

    it('collects everything into a string if sep is not included', () => {
        expect(
            splitLikePython("a=b=c=d=e", ";", 1)
        ).toEqual(["a=b=c=d=e"]);
    })

    it('correctly limits to the number of items requested (1)', () => {
        expect(
            splitLikePython("a=b=c=d=e", "=", 1)
        ).toEqual(["a", "b=c=d=e"]);
    })

    it('correctly limits to the number of items requested (2)', () => {
        expect(
            splitLikePython("a=b=c=d=e", "=", 3)
        ).toEqual(["a", "b", "c", "d=e"]);
    })
})

describe('parseHashParameters', () => {
    it('returns an empty object for empty hash values', () => {
        expect(parseHashParameters('')).toEqual({})
    })

    it('works for the easy case', () => {
        expect(parseHashParameters('a=b&c=d&e=f')).toEqual({ a: 'b', c: 'd', e: 'f' })
    })

    it('works for a single k/v pair', () => {
        expect(parseHashParameters('a=b')).toEqual({ a: 'b' })
    })

    it('works correctly if the values contain equal signs', () => {
        expect(parseHashParameters('a=b=c')).toEqual({ a: 'b=c' })
    })
})