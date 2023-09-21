import { ProgressDetails } from "../messages";
import { ProgressReducerState } from "./reducers";

export const getTotalProgress = (state: ProgressReducerState): ProgressDetails => {
    return state.ids.reduce((acc, id, idx) => {
        return {
            numFrames: acc.numFrames + state.byId[id].numFrames,
            numFramesComplete: acc.numFramesComplete + state.byId[id].numFramesComplete
        };
    }, {numFrames: 0, numFramesComplete: 0});
}