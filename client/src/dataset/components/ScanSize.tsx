import { Field } from "formik";
import * as React from "react";
import { Button, Form, FormFieldProps } from "semantic-ui-react";

interface ScanSizeProps {
    value: string,
    minScan: number,
    maxScan: number,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const ScanSize: React.FC<ScanSizeProps> = ({ value, minScan, maxScan, setFieldValue }) => {

    const [scanSize, setScanSize] = React.useState(value.split(",").length>=minScan? value.split(","): new Array(minScan).fill(""));

    const scanRefsArray = React.useRef<FormFieldProps[]>([]);

    const onScanSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const idx = parseInt(e.target.name.split("_")[2], 10);
      scanSize[idx] = e.target.value.toString();
      setScanSize(scanSize);
      setFieldValue("scan_size", scanSize.toString());
    };

    const onCommaPress = (e: KeyboardEvent) => {
        const idx = parseInt((e.target as HTMLInputElement).name.split("_")[2], 10);
        if(e.keyCode === 188){
            idx===(scanSize.length-1)? newScanDim() : scanRefsArray.current[idx+1].focus();
        }
    }

    const newScanDim = () => {
      if(scanSize.length < maxScan) {
        scanSize.push("");
        setScanSize(scanSize);
        setFieldValue("scan_size", scanSize.toString());
      }
    }

    React.useEffect(() => {
      if(scanSize.length > minScan) {
        scanRefsArray.current[scanSize.length-1].focus();
      }
    }, [scanSize.length, minScan]);

    const delScanDim = () => {
      if(scanSize.length > minScan) {
        scanSize.pop();
        setScanSize(scanSize);
        setFieldValue("scan_size", scanSize.toString());
      }
    }

    return (
      <>
        <Form.Group>
          {scanSize.map((val, idx) => {
            const scanRef = (ref:FormFieldProps) => { scanRefsArray.current[idx] = ref; }
            return <Form.Field key={idx} width={2}><Field name={"scan_size_" + idx} id={"id_scan_size_" + idx} type="number" value={val} innerRef={scanRef} onChange={onScanSizeChange} onKeyDown={onCommaPress} /></Form.Field>
          })}
          <Form.Field>
            <Button onClick={newScanDim} disabled={scanSize.length === maxScan? true: false} type="button" icon="add" title="Add dimension" basic={true} color="blue" />
            <Button onClick={delScanDim} disabled={scanSize.length === minScan? true: false} type="button" icon="minus" title="Remove dimension" basic={true} color="blue" />
          </Form.Field>
        </Form.Group>
      </>
    );
}

export default ScanSize;
