import { ErrorMessage } from "formik";
import * as React from "react";
import { Form, Input } from "semantic-ui-react";
import { frameCalcForOffset, parseNumListProduct } from "../helpers";
import TupleInput from "./TupleInput";

interface ReshapeProps {
    navShape: string,
    sigShape: string,
    syncOffset: number,
    imageCount?: number,
    hideInfo?: boolean,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const Reshape: React.FC<ReshapeProps> = ({ navShape, sigShape, syncOffset, imageCount=0, hideInfo=false, setFieldValue }) => {

    const reshapedNavShape = navShape !== undefined ? navShape : "0";
    const reshapedSigShape = sigShape !== undefined ? sigShape : "0";

    const navShapeProduct = parseNumListProduct(navShape);
    const [offsetValue, setOffset] = React.useState(syncOffset.toString());
    const offsetVal = parseInt(offsetValue, 10);

    React.useEffect(() => {
        setOffset(syncOffset.toString());
      }, [syncOffset]);

    const {framesSkippedStart, framesIgnoredEnd, framesInsertedStart, framesInsertedEnd} = frameCalcForOffset(offsetVal, navShapeProduct, imageCount);
    
    const handleOffsetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { value } = e.target;
        setOffset(value);
        setFieldValue("sync_offset", parseInt(value, 10));
    };

    return (
        <div style={{paddingBottom: 5}}>
            <Form.Field>
                <label htmlFor="id_nav_shape_0">Nav Shape:</label>
                <div style={{ color: 'red'}}><ErrorMessage name="nav_shape" /></div>
                <TupleInput value={reshapedNavShape} minLen={2} maxLen={2} fieldName="nav_shape" setFieldValue={setFieldValue} />
                <label htmlFor="id_sig_shape_0">Sig Shape:</label>
                <div style={{ color: 'red'}}><ErrorMessage name="sig_shape" /></div>
                <TupleInput value={reshapedSigShape} minLen={2} maxLen={2} fieldName="sig_shape" setFieldValue={setFieldValue} />
            </Form.Field>
            <Form.Field width={4}>
                <label htmlFor="id_sync_offset">Sync Offset (frames):</label>
                <div style={{ color: 'red'}}><ErrorMessage name="sync_offset" /></div>
                <Input type="number" required={true} name="sync_offset" id="id_sync_offset" value={offsetValue} onChange={handleOffsetChange} />
            </Form.Field>
            <Form.Field hidden={hideInfo}>
                <label>Number of frames skipped at the beginning: {framesSkippedStart}</label>
                <label>Number of blank frames inserted at the beginning: {framesInsertedStart}</label>
                <label>Number of frames ignored at the end: {framesIgnoredEnd}</label>
                <label>Number of blank frames inserted at the end: {framesInsertedEnd}</label>
            </Form.Field>
        </div>
    );

}

export default Reshape;