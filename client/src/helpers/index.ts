import * as _ from "lodash";
import { Dispatch } from "redux";
import { AllActions } from "../actions";
import { dispatchGenericError } from "../errors/helpers";

export const assertNotReached = (message: string): never => {
    throw new Error(message);
}

export const defaultDebounce = <T extends (...args: any[]) => any>(fn: T, delay = 50) => (
    _.debounce(fn, delay, { maxWait: delay })
);

export const getEnumValues = <E extends object>(e: E): Array<keyof E> => (
    Object.keys(e) as Array<keyof E>
);

export const writeClipboard = (contents: string, dispatch: Dispatch<AllActions>): void => {
    navigator.clipboard.writeText(contents).catch(() => dispatchGenericError("could not write to clipboard", dispatch));
}
