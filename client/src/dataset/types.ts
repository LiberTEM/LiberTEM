import { ById } from "../helpers/reducerHelpers";
import { DatasetState } from "../messages";

export type DatasetsState = ById<DatasetState>;

export interface OpenDatasetState {
    formVisible: boolean,
    formPath?: string,
}

export interface OpenFormProps<P> {
    onSubmit: (params: P) => void
    onCancel: () => void,
    path: string,
}