import { Dispatch } from "redux";
import { v4 as uuid } from 'uuid';
import { AllActions } from "../actions";
import * as errorActions from "./actions"

export const dispatchGenericError = (msg: string, dispatch: Dispatch<AllActions>): void => {
    const id = uuid();
    const timestamp = Date.now();
    dispatch(errorActions.Actions.generic(id, "could not write to clipboard", timestamp));
}