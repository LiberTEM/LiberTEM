import { beforeEach, describe, expect, it } from "vitest";
import { getUrlAction, parseHashParameters, splitLikePython } from "../helpers";

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

describe('getUrlAction', () => {
    beforeEach(() => {
        window.location.hash = "";
    });

    it('returns URLActionNone if no action is given in the URL', () => {
        window.location.hash = "";
        expect(getUrlAction()).toEqual({'action': 'none'});
    });

    it('works ok if we pass in something sensible', () => {
        window.location.hash = "action=open&path=/stuff";
        expect(getUrlAction()).toEqual({'action': 'open', 'path': '/stuff'});
    });

    it('correctly decodes URI components', () => {
        window.location.hash = "action=open&path=/stuff%20stuff";
        expect(getUrlAction()).toEqual({'action': 'open', 'path': '/stuff stuff'});
    });

    it('returns an error if an invalid action was given', () => {
        window.location.hash = "#action=something";
        expect(getUrlAction()).toEqual({
            'action': 'error',
            'msg': 'Unknown action specified in URL: "something"',
        });
    });
    
    it('returns an error if an invalid path is given to the open action', () => {
        window.location.hash = "#action=open&path=";
        expect(getUrlAction()).toEqual({
            'action': 'error',
            'msg': 'Invalid path given in URL: ""',
        });
    });
})