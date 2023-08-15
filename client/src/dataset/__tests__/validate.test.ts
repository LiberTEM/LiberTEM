
import { test, expect, describe } from 'vitest';

import { ErrorObject } from "ajv";
import { framesInfoAfterOffsetCorrection, productOfShapeInCommaSeparatedString } from "../helpers";
import { convertErrors } from "../validate";

describe('convertErrors', () => {
    test('properly removes the dot and brackets', () => {
        const inputErrors: ErrorObject[] = [
            {
                "keyword": "minimum",
                "instancePath": ".nav_shape[1]",
                "schemaPath": "#/properties/nav_shape/items/minimum",
                "params": { "comparison": ">=", "limit": 1, "exclusive": false },
                "message": "should be >= 1"
            }
        ];
        const res = convertErrors(inputErrors);
        expect(res).toEqual({ "nav_shape": "should be >= 1" })
    });

    test('handles inputs without brackets', () => {
        const inputErrors: ErrorObject[] = [
            {
                "keyword": "minimum",
                "instancePath": ".nav_shape",
                "schemaPath": "#/properties/nav_shape/items/minimum",
                "params": { "comparison": ">=", "limit": 1, "exclusive": false },
                "message": "should be >= 1"
            }
        ];
        const res = convertErrors(inputErrors);
        expect(res).toEqual({ "nav_shape": "should be >= 1" })
    });
});

describe('check frames with offset', () => {
    test('properly handles positive offset', () => {

        const navShapeProduct = productOfShapeInCommaSeparatedString("8,8");
        const imageCount = 64;
        const offsetVal = 8;

        const res = framesInfoAfterOffsetCorrection(offsetVal, navShapeProduct, imageCount);
    
        expect(res).toEqual({
            framesSkippedStart: 8,
            framesIgnoredEnd: 0,
            framesInsertedStart: 0,
            framesInsertedEnd: 8
        });
    });

    test('properly handles negative offset', () => {

        const navShapeProduct = productOfShapeInCommaSeparatedString("8,8");
        const imageCount = 64;
        const offsetVal = -8;

        const res = framesInfoAfterOffsetCorrection(offsetVal, navShapeProduct, imageCount);
    
        expect(res).toEqual({
            framesSkippedStart: 0,
            framesIgnoredEnd: 8,
            framesInsertedStart: 8,
            framesInsertedEnd: 0
        });
    });

    test('properly handles missing frames', () => {

        const navShapeProduct = productOfShapeInCommaSeparatedString("10,8");
        const imageCount = 64;
        const offsetVal = 0;

        const res = framesInfoAfterOffsetCorrection(offsetVal, navShapeProduct, imageCount);
    
        expect(res).toEqual({
            framesSkippedStart: 0,
            framesIgnoredEnd: 0,
            framesInsertedStart: 0,
            framesInsertedEnd: 16
        });
    });
});