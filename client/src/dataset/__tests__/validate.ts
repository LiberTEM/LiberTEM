import { ErrorObject } from "ajv";
import { convertErrors } from "../validate";

describe('convertErrors', () => {
    it('properly removes the dot and brackets', () => {
        const inputErrors: ErrorObject[] = [
            {
                "keyword": "minimum",
                "dataPath": ".scan_size[1]",
                "schemaPath": "#/properties/scan_size/items/minimum",
                "params": { "comparison": ">=", "limit": 1, "exclusive": false },
                "message": "should be >= 1"
            }
        ];
        const res = convertErrors(inputErrors);
        expect(res).toEqual({ "scan_size": "should be >= 1" })
    });

    it('handles inputs without brackets', () => {
        const inputErrors: ErrorObject[] = [
            {
                "keyword": "minimum",
                "dataPath": ".scan_size",
                "schemaPath": "#/properties/scan_size/items/minimum",
                "params": { "comparison": ">=", "limit": 1, "exclusive": false },
                "message": "should be >= 1"
            }
        ];
        const res = convertErrors(inputErrors);
        expect(res).toEqual({ "scan_size": "should be >= 1" })
    });
});