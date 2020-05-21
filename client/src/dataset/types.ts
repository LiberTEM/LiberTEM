import { ById } from "../helpers/reducerHelpers";
import { DatasetFormInfo, DatasetFormParams, DatasetState } from "../messages";

export type DatasetsState = ById<DatasetState>;

export interface OpenDatasetState {
    busy: boolean,
    busyPath: string,
    formVisible: boolean,
    formPath: string,
    formCachedParams?: DatasetFormParams,
    formDetectedParams?: DatasetFormParams,
    formDetectedInfo?: DatasetFormInfo,
}

export interface OpenFormProps<P, Q> {
    onSubmit: (params: P) => void
    onCancel: () => void,
    path: string,
    initial?: P,
    info?: Q,
}