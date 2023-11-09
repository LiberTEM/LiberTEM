import { ProgressDetails } from "../messages";
import { ProgressReducerState } from "./reducers";

export type ProgressState = Omit<ProgressDetails, 'event'>;

export const getTotalProgress = (state: ProgressReducerState): ProgressState => state.ids.reduce((acc, id, _idx) => ({
    numFrames: acc.numFrames + state.byId[id].numFrames,
    numFramesComplete: acc.numFramesComplete + state.byId[id].numFramesComplete
}), { numFrames: 0, numFramesComplete: 0 })