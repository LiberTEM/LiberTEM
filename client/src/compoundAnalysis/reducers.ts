import { AllActions } from "../actions";
import * as analysisActions from "../analysis/actions";
import * as channelActions from '../channel/actions';
import * as datasetActions from "../dataset/actions";
import { ById, constructById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import * as compoundAnalysisActions from "./actions";
import { CompoundAnalysisState } from "./types";

export type CompoundAnalysisReducerState = ById<CompoundAnalysisState>;

const initialCompoundAnalysisState: CompoundAnalysisReducerState = {
    byId: {},
    ids: [],
}

export const compoundAnalysisReducer = (state = initialCompoundAnalysisState, action: AllActions): CompoundAnalysisReducerState => {
    switch (action.type) {
        case compoundAnalysisActions.ActionTypes.CREATED: {
            const newCompoundAnalysis = {
                doAutoStart: action.payload.autoStart,
                ...action.payload.compoundAnalysis,
            }
            return insertById(state, action.payload.compoundAnalysis.compoundAnalysis, newCompoundAnalysis);
        }
        case compoundAnalysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: CompoundAnalysisState) => r.compoundAnalysis !== action.payload.id);
        }
        case compoundAnalysisActions.ActionTypes.ENABLE_AUTOSTART: {
            return updateById(state, action.payload.compoundAnalysisId, {
                doAutoStart: true,
            })
        }
        case datasetActions.ActionTypes.DELETE: {
            return filterWithPred(state, (r: CompoundAnalysisState) => r.dataset !== action.payload.dataset);
        }
        case analysisActions.ActionTypes.CREATED: {
            const compoundAnalysis = state.byId[action.payload.compoundAnalysis];
            const newAnalyses = [...compoundAnalysis.details.analyses];
            newAnalyses[action.payload.analysisIndex] = action.payload.analysis.id;
            return updateById(state, action.payload.compoundAnalysis, {
                details: {
                    analyses: newAnalyses,
                    mainType: compoundAnalysis.details.mainType,
                }
            });
        }
        case channelActions.ActionTypes.INITIAL_STATE: {
            const compoundAnalyses = action.payload.compoundAnalyses.map(ca => ({ doAutoStart: false, ...ca }));
            return {
                byId: constructById(compoundAnalyses, ca => ca.compoundAnalysis),
                ids: compoundAnalyses.map(ca => ca.compoundAnalysis),
            };
        }
    }
    return state;
}