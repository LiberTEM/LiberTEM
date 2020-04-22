import * as React from "react";
import { useDispatch, useSelector } from 'react-redux';
import { DropdownProps, Header, Segment } from "semantic-ui-react";
import uuid from "uuid/v4";
import { assertNotReached } from "../../helpers";
import { DatasetFormParams, DatasetTypes } from '../../messages';
import { RootReducer } from "../../store";
import * as datasetActions from "../actions";
import { hasKey, isAdditionalInfo } from "../helpers";
import { OpenDatasetState } from "../types";
import BLOParamsForm from "./BLOParamsForm";
import DatasetTypeSelect from "./DatasetTypeSelect";
import EMPADParamsForm from "./EMPADParamsForm";
import FRMS6ParamsForm from "./FRMS6ParamsForm";
import HDF5ParamsForm from "./HDF5ParamsForm";
import K2ISParamsForm from "./K2ISParamsForm";
import MIBParamsForm from "./MIBParamsForm";
import RawFileParamsForm from "./RawFileParamsForm";
import SERParamsForm from "./SERParamsForm";


/**
 * Get the initial selection for the dataset type dropdown. If we have a previous
 * user selection, we use it, but only if the reset button was not pressed. If it was,
 * we use the detected dataset type, falling back to RAW.
 *
 * @param didReset flag: was the reset button pressed?
 * @param openState complete OpenDatasetState instance
 */
const getDefaultDSType = (didReset: boolean, openState: OpenDatasetState) => {
    const { formCachedParams, formDetectedParams } = openState;
    if (didReset) {
        return formDetectedParams ? formDetectedParams.type : DatasetTypes.RAW;
    }
    if (formCachedParams) {
        return formCachedParams.type;
    }
    if (formDetectedParams) {
        return formDetectedParams.type;
    }
    return DatasetTypes.RAW;
}


/**
 * Get the initial form field values. If we have previously entered values, we
 * use these, otherwise we use the detected parameters. If the reset button was
 * clicked, we use the detected params, but keep the original name field,
 *
 * @param didReset flag: was the reset button pressed?
 * @param openState complete OpenDatasetState instance
 */

 // Fix this after separating info from params
const getAdditionalInfo = (formDetectedParams: DatasetFormParams) => {
     const additionalInfo = Object.keys(formDetectedParams)
     .filter(isAdditionalInfo)
     .reduce((allInfo: object, info: string) => {
       return hasKey(formDetectedParams, info)? {...allInfo, [info]: formDetectedParams[info] } : {...allInfo};
     }, {});
     return additionalInfo;
}

const getFormInitial = (didReset: boolean, openState: OpenDatasetState) => {
    const { formCachedParams, formDetectedParams } = openState;
    if (didReset) {
        if (formDetectedParams) {
            return {
                name: formCachedParams ? formCachedParams.name : "",
                ...formDetectedParams,
            };
        }
        return undefined;
    }
    if (formCachedParams) {
        if(formDetectedParams) {
          const additionalInfo = getAdditionalInfo(formDetectedParams);
          return { ...additionalInfo, ...formCachedParams };
        }
        return formCachedParams;
    } else {
        return formDetectedParams;
    }
}

/**
 * Dispatch to specific dataset opening forms, including a selection of dataset type
 * via a dropdown.
 */
const DatasetOpen = () => {
    const dispatch = useDispatch();
    const openState = useSelector((state: RootReducer) => state.openDataset);

    const [didReset, setReset] = React.useState(false);
    const formInitial = getFormInitial(didReset, openState);
    const defaultType = getDefaultDSType(didReset, openState);
    const [datasetType, setDatasetType] = React.useState(defaultType);

    const doSetType = (e: React.SyntheticEvent, data: DropdownProps) => setDatasetType(data.value as DatasetTypes);
    // FIXME: find out how to make ts correctly correlate the types of FormComponent
    // and initial and replace the huge switch below with something like:
    // const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
    // const formComponentMap = {
    //     [DatasetTypes.HDF5]: HDF5ParamsForm,
    //     [DatasetTypes.RAW]: RawFileParamsForm,
    //     [DatasetTypes.MIB]: MIBParamsForm,
    //     [DatasetTypes.BLO]: BLOParamsForm,
    //     [DatasetTypes.K2IS]: K2ISParamsForm,
    //     [DatasetTypes.SER]: SERParamsForm,
    //     [DatasetTypes.FRMS6]: FRMS6ParamsForm,
    //     [DatasetTypes.EMPAD]: EMPADParamsForm,
    // }
    // const FormComponent = formComponentMap[datasetType];

    const renderForm = (form: React.ReactNode) => {
        return (
            <Segment>
                Type: <DatasetTypeSelect onClick={doSetType} currentType={datasetType} />
                <Header as="h2">Open: {openState.formPath}</Header>
                {form}
            </Segment>
        );
    }

    const commonParams = {
        path: openState.formPath,
        onSubmit: (params: DatasetFormParams) => {
            dispatch(datasetActions.Actions.create({
                id: uuid(),
                params,
            }))
        },
        onCancel: () => dispatch(datasetActions.Actions.cancelOpen()),
        onReset: () => {
            setReset(true);
            setDatasetType(getDefaultDSType(true, openState));
        },
    }

    switch (datasetType) {
        case DatasetTypes.HDF5: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<HDF5ParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.RAW: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<RawFileParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.MIB: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<MIBParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.BLO: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<BLOParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.K2IS: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<K2ISParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.SER: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<SERParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.FRMS6: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<FRMS6ParamsForm {...commonParams} initial={initial} />);
        }
        case DatasetTypes.EMPAD: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            return renderForm(<EMPADParamsForm {...commonParams} initial={initial} />)
        }
    }
    return assertNotReached("unknown dataset type");
}

export default DatasetOpen;
