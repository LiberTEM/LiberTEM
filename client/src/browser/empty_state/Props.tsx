import { RootReducer } from "../../store";

const EmptyStateProps = (state: RootReducer) => {
    return {
        noOfDatasets: state.datasets.ids.length,
        isVisible: state.openDataset.formVisible,
        isbusy: state.openDataset.busy,
    };
}

export default EmptyStateProps 