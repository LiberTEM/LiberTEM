import { ById } from "../helpers/reducerHelpers";
import { DatasetFormParams, DatasetState } from "../messages";

export type DatasetsState = ById<DatasetState>;

export interface OpenDatasetState {
    busy: boolean,
    busyPath: string,
    formVisible: boolean,
    formPath: string,
    formCachedParams?: DatasetFormParams,
    formDetectedParams?: DatasetFormParams,
}

export interface OpenFormProps<P> {
    onSubmit: (params: P) => void
    onCancel: () => void,
    path: string,
    initial?: P,
}