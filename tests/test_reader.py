from pknn.net.datamanager import database_reader
import os


def test_reading():
    inps = ['energy', 'dist_source_tally','dist_shield_tally','mfp','theta','fi']
    n_samples = 1000
    output = 'B'
    dir = os.path.dirname(os.path.abspath(__file__))
    Reader = database_reader(os.path.join(dir,'testfiles'), [80,100,35], 
                    inputs=inps, database_inputs=inps, Output=output, sample_per_case=n_samples)
    Reader.read_data_from_file(['0_100_0.3_100'], out_log_scale=True,out_clip_values=[1,1e20])
    assert Reader.Y.get(output).size()[0] == n_samples, f"Reading test failed, size {Reader.Y.get(output).size()}"
    assert Reader.X.get(inps[1]).size()[0] == n_samples, f"Reading test failed, size {Reader.X.get(inps[1]).size()}"

 