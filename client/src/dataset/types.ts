import { ById } from "../helpers/reducerHelpers";
import { DatasetFormParams, DatasetState } from "../messages";

export type DatasetsState = ById<DatasetState>;

export interface OpenDatasetState {
    formVisible: boolean,
    formPath?: string,
    formInitialParams?: DatasetFormParams,
}

export interface OpenFormProps<P> {
    onSubmit: (params: P) => void
    onCancel: () => void,
    path: string,
    initial?: P,
}