import * as channelActions from '../channel/actions';
import { AllActions } from "../actions";
import { ById, insertById, removeById, updateIfExists } from "../helpers/reducerHelpers";
import { ProgressDetails } from "../messages";

export type ProgressReducerState = ById<ProgressDetails>;

const initialProgressstate: ProgressReducerState = {
    byId: {},
    ids: [],
}

export const progressReducer = (state = initialProgressstate, action: AllActions): ProgressReducerState => {
    switch (action.type) {
        case channelActions.ActionTypes.JOB_PROGRESS: {
            switch (action.payload.details.event) {
                case 'start':
                    return insertById(
                        state,
                        action.payload.job,
                        action.payload.details,
                    )
                case 'end':
                    return removeById(
                        state,
                        action.payload.job
                    );
                case 'update':
                    return updateIfExists(
                        state,
                        action.payload.job,
                        action.payload.details
                    );
            }
        }
        case channelActions.ActionTypes.FINISH_JOB:
        case channelActions.ActionTypes.CANCELLED: 
        case channelActions.ActionTypes.CANCEL_JOB_FAILED: {
            return removeById(state, action.payload.job);
        }
    }
    return state;
}
